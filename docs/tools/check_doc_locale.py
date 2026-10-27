#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Check bilingual documentation pairs, language labels and cross-language links.

Advisory checks (printed, never fail): pages that have no counterpart, missing
``[English]``/``[中文]`` switch lines, and heading-count drift between a pair.

Strict checks (fail with ``--strict`` or ``DOC_CHECK_STRICT=1``): an English
page linking to a ``_zh`` page whose English counterpart exists, and a Chinese
page linking to an English page whose ``_zh`` counterpart exists.  Those links
silently drop readers into the wrong language.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_catalog  # noqa: E402

ROOT = doc_catalog.ROOT
SOURCE_DIRECTORIES = ('docs', 'examples', 'example-inputs', 'examples-python',
                      'LAMMPS', 'runtime/mtp', 'mtp_kim', 'scripts/benchmark')
EXCLUDED_PARTS = {'build', 'build_zh', 'manual-html', '__pycache__'}
LINK_RE = re.compile(
    r"!?\[([^\]]*)\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^## ", re.MULTILINE)
SWITCH_ZH_RE = re.compile(r"\[中文\]\([^)]*_zh\.md\)")
SWITCH_EN_RE = re.compile(r"\[English\]\((?![^)]*_zh\.md)[^)]*\.md\)")


def pair(source: str):
    path = ROOT / source
    stem = path.stem.removesuffix("_zh")
    return path.with_name(stem + ".md"), path.with_name(stem + "_zh.md")


def relative_links(path: Path):
    text = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", path.read_text(encoding="utf-8", errors="ignore"),
                  flags=re.M | re.S)
    # Language switch links are intentional and never count as misplaced links.
    text = re.sub(r"\[(?:English|中文)\]\([^)]*\)", "", text)
    for label, target in LINK_RE.findall(text):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path.endswith(".md"):
            continue
        # An explicit "英文"/"英文原文"/"Chinese original" label marks an
        # intentional fallback to the original-language page.
        if "英文" in label or "Chinese original" in label:
            continue
        yield target, path.parent / unquote(parsed.path)


def markdown_sources():
    """Include repository entry points and examples as well as the manual."""
    paths = set(ROOT.glob('README*.md'))
    for directory in SOURCE_DIRECTORIES:
        paths.update((ROOT / directory).rglob('*.md'))
    return sorted(path for path in paths
                  if not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        default=os.environ.get("DOC_CHECK_STRICT") == "1")
    args = parser.parse_args()

    pages = []
    for item in [*doc_catalog.categories(), *doc_catalog.entries()]:
        if item["source"] in doc_catalog.BILINGUAL_PAGES:
            continue
        en, zh = pair(item["source"])
        if (en, zh) not in pages:
            pages.append((en, zh))
    # Inspect existing translation pairs outside the catalog too. Single-language
    # tool notes need not be added to the user-facing navigation to be checked.
    sources = markdown_sources()
    for path in sources:
        if path.relative_to(ROOT).as_posix() in doc_catalog.BILINGUAL_PAGES:
            continue
        en, zh = pair(str(path.relative_to(ROOT)))
        if en.is_file() and zh.is_file() and (en, zh) not in pages:
            pages.append((en, zh))

    missing_en, missing_zh, switch_problems, heading_drift = [], [], [], []
    cross_language = []
    seen = set()

    for en, zh in pages:
        if (en, zh) in seen:
            continue
        seen.add((en, zh))
        has_en, has_zh = en.is_file(), zh.is_file()
        if has_en and not has_zh:
            missing_zh.append(str(en.relative_to(ROOT)))
        elif has_zh and not has_en:
            missing_en.append(str(zh.relative_to(ROOT)))
        if not (has_en and has_zh):
            continue
        en_text = en.read_text(encoding="utf-8", errors="ignore")
        zh_text = zh.read_text(encoding="utf-8", errors="ignore")
        if not SWITCH_ZH_RE.search(en_text):
            switch_problems.append(f"{en.relative_to(ROOT)}: missing [中文](..._zh.md)")
        if not SWITCH_EN_RE.search(zh_text):
            switch_problems.append(f"{zh.relative_to(ROOT)}: missing [English](....md)")
        en_headings = len(HEADING_RE.findall(en_text))
        zh_headings = len(HEADING_RE.findall(zh_text))
        if en_headings != zh_headings:
            heading_drift.append(
                f"{en.relative_to(ROOT)} ({en_headings}) vs {zh.relative_to(ROOT)} ({zh_headings})")

    for path in sources:
        if path.relative_to(ROOT).as_posix() in doc_catalog.BILINGUAL_PAGES:
            continue
        for target, dest in relative_links(path):
            if path.stem.endswith("_zh"):
                if target.endswith("_zh.md") or not dest.exists():
                    continue
                sibling = dest.with_name(dest.stem + "_zh.md")
                if sibling.is_file():
                    cross_language.append(
                        f"{path.relative_to(ROOT)} -> {target} (use {sibling.name})")
            else:
                if not target.endswith("_zh.md") or not dest.exists():
                    continue
                english = dest.with_name(dest.stem.removesuffix("_zh") + ".md")
                if english.is_file():
                    cross_language.append(
                        f"{path.relative_to(ROOT)} -> {target} (use {english.name})")

    print(f"Bilingual pages: {len(seen)} pairs inspected")
    if missing_zh:
        print(f"\nEnglish-only pages, Chinese readers see the English original ({len(missing_zh)}):")
        for item in missing_zh:
            print(f"  - {item}")
    if missing_en:
        print(f"\nChinese-only pages, English readers see the Chinese original ({len(missing_en)}):")
        for item in missing_en:
            print(f"  - {item}")
    if switch_problems:
        print(f"\nMissing language switch lines ({len(switch_problems)}):")
        for item in switch_problems:
            print(f"  - {item}")
    if heading_drift:
        print(f"\nHeading-count drift between pairs ({len(heading_drift)}):")
        for item in heading_drift:
            print(f"  - {item}")
    if cross_language:
        print(f"\nCross-language links that bypass an existing translation ({len(cross_language)}):")
        for item in cross_language:
            print(f"  - {item}")

    if not (missing_zh or missing_en or switch_problems or heading_drift or cross_language):
        print("Bilingual structure and language links are consistent.")
        return 0
    if args.strict and cross_language:
        print("\nStrict mode: point each link at the matching language page.")
        return 1
    if args.strict:
        print("\nStrict mode: only cross-language links are blocking; other items are advisory.")
    else:
        print("\nAdvisory mode: set DOC_CHECK_STRICT=1 to fail on cross-language links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
