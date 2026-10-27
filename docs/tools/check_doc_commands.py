#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Compare the registered CLI commands/options with the command reference.

The expected surface is derived from ``apps/*.cpp`` (see
``doc_source_scan.py``); the documented surface comes from
``docs/reference/commands.md`` and ``commands_zh.md``.  Lines in
``doc_check_allowlist.txt`` record reviewed exceptions.

Exit code is 0 unless ``--strict`` (or ``DOC_CHECK_STRICT=1``) is set and a
command, option or alias differs without an allowlisted exception.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_source_scan as scan  # noqa: E402

DOCUMENTS = {
    "en": scan.ROOT / "docs/reference/commands.md",
    "zh": scan.ROOT / "docs/reference/commands_zh.md",
}
ALLOWLIST = Path(__file__).resolve().parent / "doc_check_allowlist.txt"


def load_allowlist(path: Path):
    entries = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                entries.add(line)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        default=os.environ.get("DOC_CHECK_STRICT") == "1")
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    args = parser.parse_args()

    allowed = load_allowlist(args.allowlist)
    code = scan.command_options()
    canonical, aliases = scan.registrations()
    alias_map = {}
    for alias, function in aliases:
        name = canonical.get(function)
        if name and alias != name:
            alias_map.setdefault(name, set()).add(alias)

    missing_commands = []
    missing_options = []
    extra_options = []
    alias_problems = []

    for language, path in DOCUMENTS.items():
        if not path.is_file():
            print(f"ERROR: {path.relative_to(scan.ROOT)} not found")
            return 1
        documented = scan.documented_commands(path)
        for name, options in sorted(code.items()):
            if name not in documented:
                if name not in allowed and f"{name}:-" not in allowed:
                    missing_commands.append(f"{language}: {name}")
                continue
            documented_options = documented[name]["options"]
            for option in sorted(options - documented_options):
                if option in allowed or f"{name}:{option}" in allowed:
                    continue
                missing_options.append(f"{language}: {name} --{option}")
            for option in sorted(documented_options - options):
                if f"{name}:!{option}" in allowed:
                    continue
                extra_options.append(f"{language}: {name} --{option}")
            expected = alias_map.get(name, set())
            if expected:
                documented_aliases = documented[name]["aliases"]
                lost = expected - documented_aliases
                if lost:
                    alias_problems.append(
                        f"{language}: {name} missing aliases {sorted(lost)}")
        for name in sorted(set(documented) - set(code)):
            extra_options.append(f"{language}: undocumented command section {name}")

    total = len(code)
    print(f"CLI surface: {total} commands derived from apps/, "
          f"{len(DOCUMENTS)} command references checked")
    if not (missing_commands or missing_options or extra_options or alias_problems):
        print("Command reference matches the registered CLI surface.")
        return 0

    for title, items in (("Missing command sections", missing_commands),
                         ("Options in code but not documented", missing_options),
                         ("Options documented but not registered", extra_options),
                         ("Alias labels out of date", alias_problems)):
        if not items:
            continue
        print(f"\n{title} ({len(items)}):")
        for item in items:
            print(f"  - {item}")

    if args.strict:
        blocking = bool(missing_commands or missing_options or extra_options or alias_problems)
        if blocking:
            print("\nStrict mode: update the command reference or the allowlist.")
            return 1
    else:
        print("\nAdvisory mode: set DOC_CHECK_STRICT=1 to fail on command/option/alias differences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
