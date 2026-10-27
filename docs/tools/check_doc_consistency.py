#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Check consistency between English and Chinese command-reference docs.

Verifies that:
Verifies matching command names and option sets in docs/reference/commands.md
and docs/reference/commands_zh.md.

Exit code 0 = consistent, 1 = differences found.
"""

import re
import sys
from pathlib import Path

DOC_EN = Path("docs/reference/commands.md")
DOC_ZH = Path("docs/reference/commands_zh.md")


def extract_sections(text: str):
    """Return dict mapping command-name → section text."""
    sections = re.split(r"(?=^### )", text, flags=re.MULTILINE)
    result = {}
    for sec in sections:
        m = re.search(r"^### `([^`]+)`", sec)
        if m:
            result[m.group(1)] = sec
    return result


def extract_options(text: str):
    """Return set of long-option names (without --) from a command section."""
    opts = set()
    for m in re.finditer(r"^\| `--([a-z][a-z0-9_-]+)`", text, re.MULTILINE):
        opts.add(m.group(1))
    return opts


def main():
    if not DOC_EN.exists():
        print(f"ERROR: {DOC_EN} not found")
        sys.exit(1)
    if not DOC_ZH.exists():
        print(f"ERROR: {DOC_ZH} not found")
        sys.exit(1)

    en_sections = extract_sections(DOC_EN.read_text())
    zh_sections = extract_sections(DOC_ZH.read_text())

    en_commands = set(en_sections.keys())
    zh_commands = set(zh_sections.keys())

    errors = []

    # ── Check command coverage ────────────────────────────────────────
    en_only = en_commands - zh_commands
    zh_only = zh_commands - en_commands
    if en_only:
        errors.append(f"Commands in EN only: {sorted(en_only)}")
    if zh_only:
        errors.append(f"Commands in ZH only: {sorted(zh_only)}")

    # ── Check option consistency per command ──────────────────────────
    for cmd in sorted(en_commands & zh_commands):
        en_opts = extract_options(en_sections[cmd])
        zh_opts = extract_options(zh_sections[cmd])
        en_only_opts = en_opts - zh_opts
        zh_only_opts = zh_opts - en_opts
        if en_only_opts:
            errors.append(f"{cmd}: options in EN but missing in ZH: {sorted(en_only_opts)}")
        if zh_only_opts:
            errors.append(f"{cmd}: options in ZH but missing in EN: {sorted(zh_only_opts)}")

    # ── Report ────────────────────────────────────────────────────────
    if errors:
        print("❌ Doc consistency issues found:\n")
        for e in errors:
            print(f"  • {e}")
        print(f"\n{len(errors)} issue(s) found.")
        sys.exit(1)
    else:
        print(f"✅ EN/ZH consistent: {len(en_commands)} commands, "
              f"all option sets match.")
        sys.exit(0)


if __name__ == "__main__":
    main()
