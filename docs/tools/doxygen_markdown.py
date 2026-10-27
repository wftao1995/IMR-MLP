#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Adapt repository Markdown math, page links and attachments for Doxygen."""
from functools import lru_cache
from hashlib import sha1
from html import escape, unescape
from pathlib import Path
import re
import sys
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r'(!?\[[^\]]*\]\()<?([^\s)>]+)>?((?:\s+"[^"]*")?\))')
FENCE = re.compile(r'(^```[^\n]*\n.*?^```[^\n]*$|^~~~[^\n]*\n.*?^~~~[^\n]*$)', re.M | re.S)
HEADING = re.compile(r'^(#{1,6}) +([^\n]+)$', re.M)


def prose_parts(text):
    return FENCE.split(text)


def local_target(source, target):
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    path = Path(unquote(parsed.path))
    destination = (source.parent / path).resolve() if parsed.path else source.resolve()
    # Workstation artifacts are evidence paths, not published documentation.
    if path.is_absolute() or not destination.is_relative_to(ROOT) or any(
            destination.is_relative_to(ROOT / directory)
            for directory in ('build', 'docs/build', 'docs/build_zh')):
        return None
    return destination


def preferred_page(path, language):
    if path.is_dir():
        path = path / 'README.md'
    if path.suffix == '.md':
        stem = path.stem.removesuffix('_zh')
        preferred = path.with_name(stem + ('_zh' if language == 'zh' else '') + '.md')
        if preferred.is_file():
            return preferred
    return path


@lru_cache(maxsize=None)
def page_id(path):
    heading = next((m for i, part in enumerate(prose_parts(path.read_text())) if not i % 2
                    for m in HEADING.finditer(part) if m[1] == '#'), None)
    if not heading:
        return None
    explicit = re.search(r'\{#([^}]+)\}', heading[2])
    if explicit:
        return explicit[1]
    rel = path.resolve().relative_to(ROOT).as_posix().removesuffix('.md').removesuffix('_zh')
    return 'doc_' + re.sub(r'[^A-Za-z0-9_]', '_', rel)


def heading_slug(title):
    title = re.sub(r'\{#[^}]+\}', '', title).strip().rstrip('#').strip()
    title = re.sub(r'!?\[([^\]]+)\]\([^)]*\)', r'\1', title)
    title = re.sub(r'<[^>]*>', '', unescape(title)).lower()
    return re.sub(r'\s', '-', re.sub(r'[^\w\s-]', '', title))


def section_id(identifier, slug):
    # Namespace section IDs by page; repeated headings occur throughout the guides.
    return identifier + '_section_' + sha1(slug.encode()).hexdigest()[:12]


@lru_cache(maxsize=None)
def sections(path):
    identifier = page_id(path)
    result, counts = {}, {}
    for i, part in enumerate(prose_parts(path.read_text())):
        if i % 2:
            continue
        for heading in HEADING.finditer(part):
            slug = heading_slug(heading[2])
            count = counts.get(slug, 0)
            counts[slug] = count + 1
            slug += f'-{count}' if count else ''
            explicit = re.search(r'\{#([^}]+)\}', heading[2])
            anchor = explicit[1] if explicit else (
                identifier if heading[1] == '#' else section_id(identifier, slug))
            result[slug] = anchor
            if explicit:
                result[anchor] = anchor
    return result


def render(source, language, pages=None, assets=None):
    source = source.resolve()
    identifier = page_id(source)
    counts = {}

    def heading(match):
        slug = heading_slug(match[2])
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        slug += f'-{count}' if count else ''
        if re.search(r'\{#[^}]+\}', match[2]):
            return match[0]
        anchor = identifier if match[1] == '#' else section_id(identifier, slug)
        return match[0] + ' {#' + anchor + '}'

    def target_url(target, label=''):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            return target
        if parsed.path.endswith('.html'):
            manual_page = (source.parent / unquote(parsed.path)).resolve()
            manual_root = ROOT / 'docs/manual-html'
            if manual_page.is_relative_to(manual_root):
                return '../../manual-html/' + manual_page.relative_to(manual_root).as_posix() + (
                    '#' + parsed.fragment if parsed.fragment else '')
        destination = local_target(source, target)
        if destination is None:
            return None
        if parsed.path.endswith('.html'):
            return target
        # A language selector expresses an explicit destination. Do not turn it
        # into a self-link by applying the normal same-language fallback.
        translation = {'English': 'en', '中文': 'zh', '简体中文': 'zh'}.get(label)
        destination = preferred_page(destination, translation or language)
        if destination.suffix == '.md' and destination.is_file():
            if pages is not None and not translation:
                pages.add(destination)
            anchor = page_id(destination)
            if parsed.fragment:
                anchor = sections(destination).get(unquote(parsed.fragment))
                if not anchor:
                    raise ValueError(f'{source.relative_to(ROOT)}: unknown section in {target}')
            if translation:
                filename = 'index' if page_id(destination) in ('mainpage', 'index') else page_id(destination)
                directory = 'build_zh' if translation == 'zh' else 'build'
                fragment = '#' + anchor if parsed.fragment else ''
                return f'../../{directory}/html/{filename}.html{fragment}'
            if anchor in ('mainpage', 'index'):
                return 'index.html'
            if anchor:
                return '@ref ' + anchor
        if destination.is_file():
            if assets is not None:
                assets.add(destination)
            return quote('assets/' + destination.relative_to(ROOT).as_posix())
        raise ValueError(f'{source.relative_to(ROOT)}: missing local target {target}')

    def link(match):
        # Only the catalog roots establish page parents; incidental cross-links
        # stay references. This keeps all published pages in six ordered trees.
        if (ROOT / 'docs/catalog.json').is_file():
            from doc_catalog import children
            destination = local_target(source, match[2])
            label = match[1].removeprefix('[').removesuffix('](')
            if destination and label not in ('English', '中文', '简体中文'):
                destination = preferred_page(destination, language)
                if destination in children(source, language):
                    return '\\subpage ' + page_id(destination) + ' "' + label.replace('"', '') + '"'
        target = target_url(match[2], match[1].removeprefix('[').removesuffix(']('))
        if target is None:
            return match[1].lstrip('!')[1:-2] + ' (`' + match[2] + '`)'
        return match[1] + target + match[3]

    def reference(match):
        if match[2] == '@ref':
            return match[0]
        target = target_url(match[2])
        return match[1] + ': ' + (target or match[2]) + match[3]

    def convert(text):
        text = HEADING.sub(heading, text)
        text = LINK.sub(link, text)
        text = re.sub(r'^(\[[^\]\n]+\]):\s*<?([^\s>]+)>?(.*)$', reference, text, flags=re.M)
        # Doxygen can misparse dot-prefixed inline code next to Chinese text.
        text = re.sub(r'`(\.[^`\n]+)`', lambda m: ' <code>' + escape(m[1]) + '</code> ', text)
        # Doxygen's emphasis parser does not consistently recognize Chinese
        # punctuation around **...**. Leave inline code and formulas verbatim.
        protected = []
        def protect(match):
            protected.append(match[0])
            return f'\x02INLINE{len(protected) - 1}\x03'
        text = re.sub(r'`+[^`\n]*`+|(?:@|\\)f\$.*?(?:@|\\)f\$|(?:@|\\)f\[.*?(?:@|\\)f\]'
                      r'|\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', protect, text, flags=re.S)
        text = re.sub(r'(?<![\\*])\*\*([^*\n]+)\*\*(?!\*)', r'<strong>\1</strong>', text)
        text = re.sub(r'\x02INLINE(\d+)\x03', lambda m: protected[int(m[1])], text)
        text = text.replace(r'\[', r'\f[').replace(r'\]', r'\f]')
        text = text.replace(r'\(', r'\f$').replace(r'\)', r'\f$')
        return re.sub(r'\$\$(.*?)\$\$', lambda m: r'\f[' + m[1] + r'\f]', text, flags=re.S)

    # Keep shell/Python/model-file examples verbatim.
    return ''.join(part if i % 2 else convert(part)
                   for i, part in enumerate(prose_parts(source.read_text())))


if __name__ == '__main__':
    language, filename = sys.argv[1:]
    sys.stdout.write(render(Path(filename), language))
