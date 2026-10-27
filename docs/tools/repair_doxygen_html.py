#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Repair invalid index/source links emitted by Doxygen 1.18.

Only repair a link when its original target is absent and the correct destination
can be identified from the generated headings or the canonical method definition.
The full HTML check still rejects any remaining invalid targets.
"""
from html import unescape
import re
from urllib.parse import unquote, urlsplit

from check_doc_html import Page


def repair_html(directory):
    directory = directory.resolve()
    texts = {p: p.read_text() for p in directory.glob('*.html')}
    pages = {p: Page(text) for p, text in texts.items()}

    def valid(path, target):
        parsed = urlsplit(target)
        destination = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        return destination in pages and unquote(parsed.fragment) in pages[destination].ids

    for path, text in texts.items():
        if not re.fullmatch(r'(functions|globals)(?:_.*)?\.html', path.name):
            continue

        def index_link(match):
            target = match[1]
            if valid(path, target):
                return match[0]
            parsed = urlsplit(target)
            if not parsed.fragment.startswith('index_'):
                return match[0]
            filename = parsed.path or path.name
            letter = parsed.fragment.removeprefix('index_')
            anchor = {'_': 'index__5F', '~': 'index__7E'}.get(letter, parsed.fragment)
            candidates = [filename]
            if filename == 'globals__.html':
                candidates.append('globals.html')
            if filename == 'globals.html' and letter == 'a':
                candidates.append('globals_a.html')
            for candidate in candidates:
                target = candidate + '#' + anchor
                if valid(path, target):
                    return 'href="' + target + '"'
            # Small, unsplit indexes have an alphabet bar but no section anchors.
            return match[0]

        text = re.sub(r'href="([^"]+)"', index_link, text)
        missing = {urlsplit(target).fragment for _, target in Page(text).links
                   if target.startswith('#index_') and not valid(path, target)}

        def item(match):
            label = unescape(re.sub(r'<[^>]*>', '', match[2])).strip().lower()
            anchor = 'index_' + label[:1]
            if anchor in missing:
                missing.remove(anchor)
                return '<li id="' + anchor + '">' + match[2] + '</li>'
            return match[0]

        texts[path] = re.sub(r'(<li>)(.*?)</li>', item, text, flags=re.S)

    for path, text in list(texts.items()):
        if not path.name.startswith(('class', 'struct')):
            continue

        def definition(match):
            prefix, paragraph = match[1], match[2]
            links = re.findall(r'href="([^"]+_source\.html#[^"]+)"', paragraph)
            if not links or all(valid(path, target) for target in links):
                return match[0]
            # A using-declaration can give an inherited method the derived line
            # number and the base filename. Use the base method's own definition.
            block = prefix.rsplit('<div class="memitem">', 1)[-1]
            method = re.search(r'<td class="memname">.*?<a[^>]*href="([^"]+)"', block, re.S)
            if method:
                parsed = urlsplit(method[1])
                canonical = texts.get(path.parent / parsed.path, '')
                marker = 'id="' + parsed.fragment + '"'
                if parsed.fragment and marker in canonical:
                    member = canonical.split(marker, 1)[1].split('<h2 class="memtitle">', 2)
                    if len(member) > 1:
                        body = member[1]
                        replacement = re.search(r'<p class="definition">.*?</p>', body, re.S)
                        if replacement and all(valid(path, t) for t in re.findall(
                                r'href="([^"]+_source\.html#[^"]+)"', replacement[0])):
                            return prefix + replacement[0]
            return match[0]

        texts[path] = re.sub(r'(<div class="memitem">.*?)(<p class="definition">.*?</p>)',
                             definition, text, flags=re.S)

    for path, text in texts.items():
        if text != path.read_text():
            path.write_text(text)
