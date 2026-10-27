#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Check generated HTML files, assets and section anchors without a web server."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.ids = set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        if tag == 'a' and attrs.get('name'):
            self.ids.add(attrs['name'])
        for key in ('href', 'src'):
            if attrs.get(key):
                self.links.append((self.getpos()[0], attrs[key]))


def check_html(directory, site_root=None):
    directory = directory.resolve()
    site_root = site_root.resolve() if site_root else directory
    pages = {path: Page(path.read_text()) for path in directory.rglob('*.html')}
    errors = []
    if directory / 'index.html' not in pages:
        return [f'{directory}: missing index.html'], len(pages)
    links = {path: page.links for path, page in pages.items()}
    # The sidebar, menu and search results construct their links from JavaScript.
    for path in directory.rglob('*.js'):
        text = path.read_text()
        links[path] = [(text.count('\n', 0, match.start()) + 1, match[2])
                       for match in re.finditer(
                           r'''(["'])([^"'\n]+\.html(?:#[^"'\n]*)?)\1''', text)]
    for path, targets in sorted(links.items()):
        for line, target in targets:
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            location = f'{path.relative_to(directory)}:{line}'
            if not destination.is_relative_to(site_root) or not destination.exists():
                errors.append(f'{location}: missing file: {target}')
            elif parsed.fragment and destination.suffix == '.html':
                if destination not in pages:
                    pages[destination] = Page(destination.read_text())
                if unquote(parsed.fragment) not in pages[destination].ids:
                    errors.append(f'{location}: missing anchor: {target}')
    return errors, len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--site-root', type=Path, help='common publication root for bilingual links')
    args = parser.parse_args()
    failed = False
    for directory in args.directories:
        errors, count = check_html(directory, args.site_root)
        if errors:
            print('\n'.join(errors))
            print(f'{directory}: {len(errors)} broken links in {count} HTML pages')
            failed = True
        else:
            print(f'{directory}: local files and anchors OK in {count} HTML pages')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
