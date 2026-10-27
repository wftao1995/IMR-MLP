#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Compile the TeX in generated HTML with the site's local MathJax runtime."""
import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MATH = re.compile(r'\\\((.*?)\\\)|\\\[(.*?)\\\]', re.S)


class Formulas(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.skipped = []
        self.formulas = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag in ('pre', 'code', 'script', 'style', 'textarea', 'noscript'):
            self.skipped.append(tag)

    def handle_endtag(self, tag):
        if self.skipped and tag == self.skipped[-1]:
            self.skipped.pop()

    def handle_data(self, data):
        if self.skipped:
            return
        for match in MATH.finditer(data):
            self.formulas.append(dict(
                line=self.getpos()[0] + data.count('\n', 0, match.start()),
                tex=match[1] if match[1] is not None else match[2],
                display=match[2] is not None))


def check_math(directories, mathjax=None):
    runtime = Path(mathjax or os.environ.get(
        'MATHJAX_DIR', ROOT / 'build/docs-deps/node_modules/mathjax/es5')).resolve()
    if not (runtime / 'node-main.js').is_file():
        raise RuntimeError('Local MathJax is missing; run make doc-deps or set MATHJAX_DIR')
    formulas, count = [], 0
    for directory in directories:
        if not (directory / 'index.html').is_file():
            raise RuntimeError(f'{directory}: missing index.html')
        for page in sorted(directory.rglob('*.html')):
            count += 1
            formulas.extend(dict(file=str(page), **formula)
                            for formula in Formulas(page.read_text()).formulas)
    result = subprocess.run(
        ['node', str(ROOT / 'docs/tools/check_doc_math.cjs'), str(runtime)],
        input=json.dumps(formulas), text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    report.update(pages=count, formulas=len(formulas))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    report = check_math(args.directories)
    for error in report['errors']:
        print(f"{error['file']}:{error['line']}: {error['error']}\n  {error['tex']}")
    print(f"MathJax: {report['formulas']} formulas ({report['unique']} unique) "
          f"in {report['pages']} HTML pages; {len(report['errors'])} errors")
    return bool(report['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
