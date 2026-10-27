#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Check local Markdown file links; HTML/Doxygen targets and anchors are excluded."""
import argparse
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='files/directories, relative to repository root')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sources = [root/p for p in args.paths] if args.paths else [
        *root.glob('README*.md'), root/'CONTRIBUTING.md',
        root/'THIRD_PARTY_NOTICES.md',
        root/'docs', root/'examples', root/'example-inputs', root/'examples-python',
        root/'LAMMPS', root/'runtime/mtp', root/'mtp_kim', root/'scripts/benchmark']
    files = set()
    for source in sources:
        if not source.exists():
            parser.error(f'missing source: {source}')
        files.update(source.rglob('*.md') if source.is_dir() else [source])
    errors = []
    checked = 0
    for source in sorted(files):
        if any(part in {'build', 'build_zh', 'manual-html', '.git', '__pycache__'} for part in source.relative_to(root).parts):
            continue
        checked += 1
        # Link-like text inside fenced code blocks is not a rendered Markdown link.
        text = re.sub(r'^(```|~~~).*?^\1[^\n]*$', '', source.read_text(), flags=re.M | re.S)
        targets = re.findall(r'!?\[[^\]]*\]\(<?([^\s)>]+)>?(?:\s+"[^"]*")?\)', text)
        targets += re.findall(r'^\[[^\]\n]+\]:\s*<?([^\s>]+)>?', text, re.M)
        for target in targets:
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.endswith('.html'):
                continue
            dest = source.parent/unquote(parsed.path)
            if not dest.exists():
                errors.append(f'{source.relative_to(root)}: {target}')
    if errors:
        print('\n'.join(errors))
        print(f'{len(errors)} missing local targets in {checked} Markdown files')
        return 1
    print(f'Local file links OK in {checked} Markdown files (anchors, generated HTML and external URLs not checked)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
