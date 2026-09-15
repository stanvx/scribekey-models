from __future__ import annotations

import argparse
from pathlib import Path

from scribekey_models.catalog import GENERATED_DIR, export_generated, validate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scribekey-models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate catalogues and committed generated files")
    generate = subparsers.add_parser("generate", help="Generate runtime catalogues")
    generate.add_argument("--check", action="store_true", help="Fail if generated files are stale")
    generate.add_argument(
        "--output-dir",
        type=Path,
        default=GENERATED_DIR,
        help="Directory to write or check runtime catalogues",
    )
    return parser


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

    issues = validate()
    if issues:
        for issue in issues:
            print(f"{issue.source}: {issue.message}")
        raise SystemExit(1)
    print("Catalogue validation passed")

