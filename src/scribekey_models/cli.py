from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import yaml

from scribekey_models.catalog import (
    GENERATED_DIR,
    ROOT,
    export_generated,
    generate_cleanup_catalog,
    generate_diarization_manifest,
    generate_speech_catalog,
    load_all_releases_data,
    validate,
)
from scribekey_models.health import check_channel_sources, write_report
from scribekey_models.mirror import (
    check_mirror_configuration,
    plan_release_mirror,
)
from scribekey_models.promotion import (
    create_release_manifest,
    promote,
)
from scribekey_models.signing import (
    DEFAULT_KEY_ID,
    export_private_key_pem,
    export_public_key_pem,
    generate_keypair,
    load_private_key,
    load_public_key,
    sign_file,
    verify_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scribekey-models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate
    subparsers.add_parser(
        "validate",
        help="Validate catalogues, channels, releases, signatures, and guardrails",
    )

    # generate
    generate = subparsers.add_parser(
        "generate",
        help="Generate runtime catalogues and channel distribution metadata",
    )
    generate.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated files are stale",
    )
    generate.add_argument(
        "--output-dir",
        type=Path,
        default=GENERATED_DIR,
        help="Directory to write or check runtime catalogues",
    )

    # release
    release_cmd = subparsers.add_parser("release", help="Manage model releases")
    release_subs = release_cmd.add_subparsers(dest="release_command", required=True)
    rel_create = release_subs.add_parser("create", help="Create an immutable release definition")
    rel_create.add_argument("--id", required=True, help="Release ID (e.g. 2026.09.1)")
    rel_create.add_argument("--description", default="", help="Description of release")

    # promote
    promote_cmd = subparsers.add_parser("promote", help="Promote a release to QA or stable channel")
    promote_cmd.add_argument(
        "--channel",
        choices=["qa", "stable"],
        required=True,
        help="Distribution channel to update",
    )
    promote_cmd.add_argument("--release", required=True, help="Release ID to promote")
    promote_cmd.add_argument(
        "--sequence",
        type=int,
        default=None,
        help="Explicit monotonic sequence number (auto-increments if omitted)",
    )
    promote_cmd.add_argument("--notes", default="", help="Optional promotion notes")
    promote_cmd.add_argument("--key", default=None, help="Private signing key (string)")
    promote_cmd.add_argument(
        "--key-env",
        default=None,
        help="Environment variable containing private signing key",
    )
    promote_cmd.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="Path to file containing private signing key",
    )
    promote_cmd.add_argument(
        "--key-id",
        default=DEFAULT_KEY_ID,
        help="Key identifier to attach to signature",
    )

    # sign
    sign_cmd = subparsers.add_parser("sign", help="Sign distribution metadata")
    sign_cmd.add_argument("--target", type=Path, required=True, help="File or directory to sign")
    sign_cmd.add_argument("--key", default=None, help="Private signing key (string)")
    sign_cmd.add_argument(
        "--key-env",
        default=None,
        help="Environment variable containing private signing key",
    )
    sign_cmd.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="Path to file containing private signing key",
    )
    sign_cmd.add_argument(
        "--key-id",
        default=DEFAULT_KEY_ID,
        help="Key identifier to attach to signature",
    )

    # verify
    verify_cmd = subparsers.add_parser("verify", help="Verify detached signature")
    verify_cmd.add_argument("--target", type=Path, required=True, help="Target file to verify")
    verify_cmd.add_argument(
        "--sig-file",
        type=Path,
        default=None,
        help="Signature file (defaults to <target>.sig)",
    )
    verify_cmd.add_argument(
        "--public-key",
        type=Path,
        default=None,
        help="Path to public key file",
    )

    # keygen
    keygen_cmd = subparsers.add_parser("keygen", help="Generate a P-256 ECDSA keypair for release signing")
    keygen_cmd.add_argument("--out-public", type=Path, default=None, help="Path to write public key PEM")
    keygen_cmd.add_argument("--out-private", type=Path, default=None, help="Path to write private key PEM")

    # mirror
    mirror_cmd = subparsers.add_parser("mirror", help="Inspect and dry-run redistribution-cleared mirror assets")
    mirror_subs = mirror_cmd.add_subparsers(dest="mirror_command", required=True)
    m_plan = mirror_subs.add_parser("plan", help="Plan mirror assets for a release")
    m_plan.add_argument("--release", required=True, help="Release ID (e.g. 2026.09.1)")
    mirror_subs.add_parser("check", help="Check mirror configuration status")

    # health
    health_cmd = subparsers.add_parser(
        "health",
        help="Check the published channel's configured model artifact sources",
    )
    health_cmd.add_argument(
        "--channel",
        choices=["qa", "stable"],
        default="stable",
        help="Published channel to inspect",
    )
    health_cmd.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-source network timeout in seconds",
    )
    health_cmd.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path for the machine-readable JSON report",
    )

    return parser


def _resolve_private_key(args: argparse.Namespace):
    if args.key or args.key_env or args.key_file:
        return load_private_key(args.key, env_var=args.key_env, file_path=args.key_file)
    if "MODEL_RELEASE_SIGNING_KEY" in os.environ:
        return load_private_key(env_var="MODEL_RELEASE_SIGNING_KEY")
    return None


def _git_source_identity(root: Path) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            ["git", "show", "-s", "--format=%H%n%cI", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit, timestamp = completed.stdout.strip().splitlines()
        created_at = datetime.datetime.fromisoformat(timestamp).astimezone(datetime.UTC)
        return commit, created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (subprocess.SubprocessError, ValueError, OSError):
        now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return "0" * 40, now


def main() -> None:
    args = _parser().parse_args()

    if args.command == "generate":
        drift = export_generated(args.output_dir, check=args.check)
        if drift:
            for message in drift:
                print(message)
            raise SystemExit(1)
        print("Generated catalogues are current" if args.check else "Generated runtime catalogues")
        return

    if args.command == "release" and args.release_command == "create":
        root = Path.cwd()
        rel_file = root / "catalog" / "releases" / f"{args.id}.yaml"
        release_dir = root / "generated" / "releases" / args.id
        if rel_file.exists():
            print(f"Error: Release '{args.id}' already exists at {rel_file}")
            raise SystemExit(1)
        if release_dir.exists():
            print(f"Error: Immutable release snapshot already exists at {release_dir}")
            raise SystemExit(1)

        speech_str = json.dumps(generate_speech_catalog(), indent=2, ensure_ascii=False) + "\n"
        diarization_str = json.dumps(generate_diarization_manifest(), indent=2, ensure_ascii=False) + "\n"
        cleanup_str = json.dumps(generate_cleanup_catalog(), indent=2, ensure_ascii=False) + "\n"
        git_commit, _ = _git_source_identity(root)

        manifest = create_release_manifest(
            args.id,
            args.description or f"Release {args.id}",
            speech_content=speech_str,
            diarization_content=diarization_str,
            cleanup_content=cleanup_str,
            git_commit=git_commit,
        )
        rel_file.parent.mkdir(parents=True, exist_ok=True)
        rel_file.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        release_dir.mkdir(parents=True, exist_ok=False)
        (release_dir / "model_catalog.json").write_text(speech_str, encoding="utf-8")
        (release_dir / "speaker_diarization_manifest.json").write_text(
            diarization_str,
            encoding="utf-8",
        )
        (release_dir / "cleanup_model_catalog.json").write_text(cleanup_str, encoding="utf-8")
        export_generated()
        print(f"Created release '{args.id}' at {rel_file}")
        return

    if args.command == "promote":
        priv_key = _resolve_private_key(args)
        if priv_key is None:
            print(
                "Error: Promotion must be signed. Pass --key, --key-env, --key-file, "
                "or set MODEL_RELEASE_SIGNING_KEY."
            )
            raise SystemExit(1)
        result = promote(
            args.channel,
            args.release,
            sequence=args.sequence,
            notes=args.notes,
            private_key=priv_key,
            key_id=args.key_id,
        )
        print(result.message)
        if result.signature_path:
            print(f"Signed: {result.signature_path}")
        return

    if args.command == "sign":
        priv_key = _resolve_private_key(args)
        if priv_key is None:
            print("Error: No signing key provided. Pass --key, --key-env, --key-file, or set MODEL_RELEASE_SIGNING_KEY.")
            raise SystemExit(1)

        target = args.target
        targets = [target] if target.is_file() else list(target.rglob("*.json"))
        for t in targets:
            sig = sign_file(t, priv_key, key_id=args.key_id)
            print(f"Signed {t.name} -> {sig.name}")
        return

    if args.command == "verify":
        pub = load_public_key(args.public_key)
        ok, msg = verify_file(args.target, args.sig_file, public_key=pub)
        if not ok:
            print(f"Verification failed: {msg}")
            raise SystemExit(1)
        print(f"Verification succeeded for {args.target}")
        return

    if args.command == "keygen":
        priv, pub = generate_keypair()
        priv_pem = export_private_key_pem(priv)
        pub_pem = export_public_key_pem(pub)
        if args.out_public:
            args.out_public.parent.mkdir(parents=True, exist_ok=True)
            args.out_public.write_text(pub_pem, encoding="utf-8")
            print(f"Public key written to {args.out_public}")
        else:
            print("Public Key:")
            print(pub_pem)
        if args.out_private:
            args.out_private.parent.mkdir(parents=True, exist_ok=True)
            args.out_private.write_text(priv_pem, encoding="utf-8")
            print(f"Private key written to {args.out_private}")
        else:
            print("Private Key (Store as MODEL_RELEASE_SIGNING_KEY secret):")
            print(priv_pem)
        return

    if args.command == "mirror":
        if args.mirror_command == "check":
            status = check_mirror_configuration()
            print(json.dumps(status, indent=2))
            return
        if args.mirror_command == "plan":
            releases = load_all_releases_data()
            if args.release not in releases:
                print(f"Error: Release '{args.release}' not found in catalog/releases/")
                raise SystemExit(1)
            plans = plan_release_mirror(releases[args.release])
            print(f"Found {len(plans)} cleared mirror assets for release {args.release}:")
            for p in plans:
                print(f" - {p.model_id}:{p.filename} ({p.license_id}) -> {p.target_path}")
            return

    if args.command == "health":
        report = check_channel_sources(
            ROOT,
            channel=args.channel,
            timeout_seconds=args.timeout,
        )
        if args.report:
            write_report(report, args.report)
        verified_checksums = sum(check.checksum_verified for check in report.checks)
        print(
            f"Checked {len(report.checks)} sources for {report.channel}/{report.release_id}; "
            f"provider checksum metadata verified for {verified_checksums}."
        )
        if not report.healthy:
            for check in report.failures:
                print(f"{check.target.model_id}:{check.target.file_name}: {check.reason}")
            raise SystemExit(1)
        print("All configured model sources are healthy")
        return

    issues = validate()
    if issues:
        for issue in issues:
            print(f"{issue.source}: {issue.message}")
        raise SystemExit(1)
    print("Catalogue validation passed")
