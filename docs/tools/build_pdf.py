#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Compile the Markdown manual into PDF from the Doxygen LaTeX output.

`make doc-html` writes a LaTeX tree per language under docs/build*/latex.
This script turns that tree into a manual-only document: the whole Markdown
page hierarchy hangs off the mainpage (index.tex), so the body is just
`\\input{index}` followed by the index. The generated API reference chapters
(classes, files, namespaces) are never included.

Run it through `make doc-pdf`, which regenerates the LaTeX sources first.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'docs' / 'manual-pdf'
LATEX_CMD = 'xelatex'
# The full document needs more string pool than the TeX Live default.
LATEX_OPTIONS = ['-cnf-line=pool_size=20000000', '-interaction=nonstopmode']
LANGUAGES = {
    'en': dict(
        tree=ROOT / 'docs/build/latex', chapter='IMR-MLP Manual', output='imrmlp-manual-en.pdf'
    ),
    'zh': dict(
        tree=ROOT / 'docs/build_zh/latex', chapter='IMR-MLP 手册', output='imrmlp-manual-zh.pdf'
    ),
}


def strip_underscores_in_text(tree):
    """Escape `_` inside `\\text{...}`; Doxygen leaves it raw inside math.

    `@f$N_{\\text{atoms}}^{\\text{energy_pow}}@f$` becomes `\\text{energy_pow}`,
    which is invalid in math mode and aborts the formula.
    """
    pattern = re.compile(r'\\text\{([^{}]*)\}')
    changed = 0
    for path in tree.glob('*.tex'):
        text = path.read_text(encoding='utf-8', errors='replace')
        fixed = pattern.sub(
            lambda m: r'\text{' + re.sub(r'(?<!\\)_', r'\\_', m.group(1)) + '}', text
        )
        if fixed != text:
            path.write_text(fixed, encoding='utf-8')
            changed += 1
    return changed


def fix_image_widths(tree):
    """Undo Doxygen's doubled backslash in `\\includegraphics[width=0.6\\\\textwidth]`.

    The markdown form `![...](logo.png){width=60%}` reaches LaTeX as
    `width=0.6\\\\textwidth`, where `\\\\` starts a line break inside the
    optional argument and aborts the box with calc errors.
    """
    changed = 0
    for path in tree.glob('*.tex'):
        text = path.read_text(encoding='utf-8', errors='replace')
        lines = []
        for line in text.splitlines(keepends=True):
            if 'includegraphics' in line:
                line = line.replace('\\\\textwidth', '\\textwidth')
            lines.append(line)
        fixed = ''.join(lines)
        if fixed != text:
            path.write_text(fixed, encoding='utf-8')
            changed += 1
    return changed


def drop_legacy_cjk(tree):
    """Remove Doxygen's pdfTeX-era CJK wrapper; xeCJK handles CJK under XeTeX."""
    path = tree / 'refman.tex'
    text = path.read_text(encoding='utf-8', errors='replace')
    fixed = text.replace('  \\usepackage{CJKutf8}\n', '')
    fixed = re.sub(r'^\s*\\begin\{CJK\}\{[^}]*\}\{[^}]*\}\s*\n', '', fixed, flags=re.M)
    fixed = re.sub(r'^\s*\\end\{CJK\}\s*\n', '', fixed, flags=re.M)
    if fixed != text:
        path.write_text(fixed, encoding='utf-8')
        return True
    return False


def rename_chapter(tree, title):
    """Title the single manual chapter instead of Doxygen's mainpage title."""
    path = tree / 'index.tex'
    text = path.read_text(encoding='utf-8', errors='replace')
    fixed, count = re.subn(r'\\chapter\{[^}]*\}', '\\\\chapter{' + title + '}', text, count=1)
    if count and fixed != text:
        path.write_text(fixed, encoding='utf-8')
        return True
    return False


def write_manual(tree):
    """Keep the preamble and the Markdown page tree, drop the API chapters."""
    source = (tree / 'refman.tex').read_text(encoding='utf-8', errors='replace')
    marker = '%--- Begin generated contents ---'
    if marker not in source:
        raise RuntimeError(f'{tree / "refman.tex"} has no generated contents marker')
    head = source.split(marker, 1)[0]
    body = (
        '%--- Manual only: the API reference chapters are not included. ---\n'
        '\\input{index}\n'
        '  \\phantomsection\n'
        '  \\clearemptydoublepage\n'
        '  \\addcontentsline{toc}{chapter}{\\indexname}\n'
        '  \\printindex\n'
        '\\end{document}\n'
    )
    target = tree / 'refman_manual.tex'
    target.write_text(head + body, encoding='utf-8')
    return target


def needs_rerun(log):
    text = log.read_text(encoding='utf-8', errors='replace')
    return any(
        marker in text
        for marker in (
            'Rerun to get cross-references right',
            'Label(s) may have changed',
            'There were undefined references',
        )
    )


def compile_pdf(tree, passes=5):
    log = tree / 'refman_manual.log'
    for attempt in range(passes):
        result = subprocess.run(
            [LATEX_CMD, *LATEX_OPTIONS, 'refman_manual.tex'],
            cwd=tree,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        index = tree / 'refman_manual.idx'
        if attempt == 0 and index.is_file():
            subprocess.run(
                ['makeindex', '-q', 'refman_manual.idx'],
                cwd=tree,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if result.returncode and not log.is_file():
            raise RuntimeError(f'{LATEX_CMD} failed in {tree} without writing a log')
        if attempt >= 1 and not needs_rerun(log):
            break
    errors = [
        line
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines()
        if line.startswith('! ')
    ]
    if errors:
        raise RuntimeError(f'{len(errors)} LaTeX error(s) in {log}:\n' + '\n'.join(errors[:5]))
    missing = log.read_text(encoding='utf-8', errors='replace').count('Missing character')
    pdf = tree / 'refman_manual.pdf'
    if not pdf.is_file():
        raise RuntimeError(f'no PDF produced in {tree}; see {log}')
    return pdf, missing


def build(language):
    spec = LANGUAGES[language]
    tree = spec['tree']
    if not (tree / 'refman.tex').is_file():
        raise RuntimeError(f'{tree} has no refman.tex; run `make doc-html` first')
    fixes = []
    if strip_underscores_in_text(tree):
        fixes.append('math text')
    if fix_image_widths(tree):
        fixes.append('image width')
    if language == 'zh' and drop_legacy_cjk(tree):
        fixes.append('CJK wrapper')
    if rename_chapter(tree, spec['chapter']):
        fixes.append('chapter title')
    write_manual(tree)
    pdf, missing = compile_pdf(tree)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / spec['output']
    shutil.copyfile(pdf, target)
    pages = re.search(
        r'Output written on refman_manual\.pdf \((\d+) pages',
        (tree / 'refman_manual.log').read_text(encoding='utf-8', errors='replace'),
    )
    detail = f'{pages.group(1)} pages' if pages else 'pages unknown'
    if missing:
        detail += f', {missing} missing glyph(s)'
    print(
        f'{language}: {target.relative_to(ROOT)} ({detail})'
        + (f'; applied fixes: {", ".join(fixes)}' if fixes else '')
    )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--lang',
        choices=['en', 'zh', 'all'],
        default='all',
        help='language to build (default: all)',
    )
    args = parser.parse_args()
    languages = ['en', 'zh'] if args.lang == 'all' else [args.lang]
    try:
        for language in languages:
            build(language)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'PDF build failed: {error}\n')


if __name__ == '__main__':
    main()
