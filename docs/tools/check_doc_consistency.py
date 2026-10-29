#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""
Check consistency between English and Chinese command-reference docs.

Verifies that:
Verifies matching command names, option sets and option-table shape in
docs/reference/commands.md and docs/reference/commands_zh.md.

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


def extract_table_header(text: str):
    """Return the first Markdown table header row of a section as lower-case cells."""
    for line in text.splitlines():
        if line.startswith("|"):
            return [cell.strip().lower() for cell in line.strip().strip("|").split("|")]
    return None


def main():
    en_path, zh_path = DOC_EN, DOC_ZH
    if not en_path.exists():
        print(f"ERROR: {en_path} not found")
        return 1
    if not zh_path.exists():
        print(f"ERROR: {zh_path} not found")
        return 1

    en_sections = extract_sections(en_path.read_text())
    zh_sections = extract_sections(zh_path.read_text())

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

    # ── Check option-table shape per command ──────────────────────────
    # Option names alone can survive a table that lost its alias or default
    # column; compare the header rows so a dropped column is reported instead
    # of silently shrinking what readers of one language can see.
    for cmd in sorted(en_commands & zh_commands):
        en_header = extract_table_header(en_sections[cmd])
        zh_header = extract_table_header(zh_sections[cmd])
        if not en_header or not zh_header:
            continue
        if len(en_header) != len(zh_header):
            errors.append(f"{cmd}: option table has {len(en_header)} columns in EN "
                          f"but {len(zh_header)} in ZH")
            continue
        if ("default" in en_header) != ("默认值" in zh_header):
            errors.append(f"{cmd}: default-value column present in only one language")
        if ("alias" in en_header) != ("别名" in zh_header):
            errors.append(f"{cmd}: alias column present in only one language")

    # ── Report ────────────────────────────────────────────────────────
    if errors:
        print("❌ Doc consistency issues found:\n")
        for e in errors:
            print(f"  • {e}")
        print(f"\n{len(errors)} issue(s) found.")
        return 1
    else:
        print(f"✅ EN/ZH consistent: {len(en_commands)} commands, "
              f"option sets and option-table shape match.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
