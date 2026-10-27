#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Build complete, fresh Doxygen output and validate it before replacing old HTML."""
import argparse
from hashlib import sha256
from html import escape
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from check_doc_html import check_html
from doxygen_markdown import preferred_page, render, ROOT
from repair_doxygen_html import repair_html
from publish_docs import publish_directory


def copy_mathjax(html):
    source = Path(os.environ.get('MATHJAX_DIR', ROOT / 'build/docs-deps/node_modules/mathjax/es5'))
    if not (source / 'tex-svg.js').is_file():
        raise RuntimeError('Local MathJax 3 assets are missing; set MATHJAX_DIR to its es5 directory')
    shutil.copytree(source, html / 'assets/mathjax/es5')
    license_file = source.parent / 'LICENSE'
    if not license_file.is_file():
        raise RuntimeError(f'MathJax license is missing: {license_file}')
    shutil.copy2(license_file, html / 'assets/mathjax/LICENSE')


def version_assets(html):
    # Both static and lazily loaded navigation/search data must use one revision.
    # Unversioned index.js can otherwise keep obsolete page names in a browser.
    files = sorted(p for p in html.rglob('*') if p.suffix in ('.js', '.css') and p.is_file())
    digest = sha256()
    for path in files:
        digest.update(path.relative_to(html).as_posix().encode())
        digest.update(path.read_bytes())
    version = digest.hexdigest()[:16]
    for filename, assignment in (
            ('navtree.js', "script.src = scriptName+'.js';"),
            ('search/search.js', 'scriptTag.src = url;')):
        path = html / filename
        if path.is_file():
            text = path.read_text()
            replacement = (f"script.src = scriptName+'.js?v={version}';" if filename == 'navtree.js'
                           else f"scriptTag.src = url + '?v={version}';")
            path.write_text(text.replace(assignment, replacement))

    def reference(match):
        parsed = urlsplit(match[2])
        if parsed.scheme or parsed.netloc or not parsed.path.endswith(('.js', '.css')):
            return match[0]
        query = [(key, value) for key, value in parse_qsl(parsed.query) if key != 'v']
        query.append(('v', version))
        target = urlunsplit(parsed._replace(query=urlencode(query)))
        return match[1] + escape(target, quote=True) + match[3]

    for path in html.rglob('*.html'):
        text = path.read_text()
        path.write_text(re.sub(r'(<(?:script|link)\b[^>]*\b(?:src|href)=")([^"]+)(")',
                               reference, text))
    (html / 'asset-version.txt').write_text(version + '\n')
    return version


def config_values(text):
    values = {}
    for line in text.replace('\\\n', ' ').splitlines():
        match = re.match(r'^\s*(\w+)\s*(\+?=)\s*(.*)$', line)
        if match:
            tokens = shlex.split(match[3], comments=True)
            values[match[1]] = (values.get(match[1], []) if match[2] == '+=' else []) + tokens
    return values


def collect_inputs(config, language):
    values = config_values(config.read_text())
    excluded = {(config.parent / name).resolve() for name in values.get('EXCLUDE', [])}
    pages = set()
    if config.resolve().parent == ROOT / 'docs':
        from doc_catalog import sources
        pages.update(sources(language))
        pages.add(ROOT / 'docs' / ('00_mainpage_zh.md' if language == 'zh' else '00_mainpage.md'))
    for name in values['INPUT']:
        path = (config.parent / name).resolve()
        candidates = path.glob('*.md') if path.is_dir() else [path]
        pages.update(preferred_page(p, language) for p in candidates if p.suffix == '.md' and p not in excluded)
    assets, visited = set(), set()
    while pages - visited:
        source = min(pages - visited)
        render(source, language, pages, assets)
        visited.add(source)
    return pages, assets


def build(config, doxygen, output=None, validate=True):
    config = config.resolve()
    text = config.read_text()
    values = config_values(text)
    language = 'zh' if values['OUTPUT_LANGUAGE'] == ['Chinese'] else 'en'
    pages, assets = collect_inputs(config, language)
    output = output or (config.parent / values['OUTPUT_DIRECTORY'][0]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.doxygen-', dir=output) as temporary:
        staging = Path(temporary)
        # Include every linked Markdown page, including nested README files.
        extra = '\nINPUT += ' + ' '.join('"' + p.as_posix() + '"' for p in sorted(pages))
        # INPUT directories also expose both translated README files to Doxygen.
        # Keep only the selected language, even when the filter could translate
        # the other file; otherwise both inputs claim the same stable page ID.
        unselected = set()
        for name in values['INPUT']:
            path = (config.parent / name).resolve()
            candidates = path.glob('*.md') if path.is_dir() else [path]
            unselected.update(p for p in candidates
                              if p.suffix == '.md' and preferred_page(p, language) != p)
        if unselected:
            extra += '\nEXCLUDE += ' + ' '.join('"' + p.as_posix() + '"' for p in sorted(unselected))
        extra += f'\nOUTPUT_DIRECTORY = "{staging}"\n'
        extra += 'WARN_LOGFILE = "' + (staging / 'warnings.log').as_posix() + '"\n'
        subprocess.run([doxygen, '-'], input=text + extra, text=True, cwd=config.parent, check=True)
        html = staging / values.get('HTML_OUTPUT', ['html'])[0]
        for asset in sorted(assets):
            destination = html / 'assets' / asset.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset, destination)
        if values.get('USE_MATHJAX') == ['YES'] and values.get('MATHJAX_RELPATH') == ['assets/mathjax']:
            copy_mathjax(html)
        warnings = (staging / 'warnings.log').read_text()
        if warnings:
            print(warnings, end='')
        if 'unable to resolve reference' in warnings or 'multiple use of page label' in warnings:
            raise RuntimeError('Doxygen produced unresolved references or duplicate page labels')
        repair_html(html)
        if validate:
            version_assets(html)
        errors, count = check_html(html) if validate else ([], len(list(html.rglob('*.html'))))
        if errors:
            print('\n'.join(errors))
            raise RuntimeError(f'{len(errors)} broken links in {count} generated HTML pages')
        for generated in staging.iterdir():
            destination = output / generated.name
            if generated.is_dir():
                publish_directory(generated, destination)
            else:
                os.replace(generated, destination)
    print(f'{config.name}: {count} HTML pages and {len(assets)} attachments generated' + (' and validated' if validate else ''))


def build_bilingual(doxygen):
    (ROOT / 'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='docs-site-', dir=ROOT / 'build') as temporary:
        staging = Path(temporary)
        for name, config in [('build', 'Doxyfile'), ('build_zh', 'Doxyfile.zh')]:
            build(ROOT / 'docs' / config, doxygen, staging / name, validate=False)
        for name in ('build', 'build_zh'):
            html = staging / name / 'html'
            version_assets(html)
            errors, count = check_html(html, staging)
            if errors:
                raise RuntimeError('\n'.join(errors[:30]) + f'\n{len(errors)} broken bilingual links')
            print(f'{name}: {count} HTML pages and bilingual links validated')
        for name in ('build', 'build_zh'):
            publish_directory(staging / name, ROOT / 'docs' / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, nargs='?')
    parser.add_argument('--all', action='store_true', help='build and check both languages as one site')
    parser.add_argument('--doxygen', default='doxygen')
    args = parser.parse_args()
    try:
        if args.all:
            build_bilingual(args.doxygen)
        elif args.config:
            build(args.config, args.doxygen)
        else:
            parser.error('provide a config file or --all')
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Documentation build failed: {error}\n')


if __name__ == '__main__':
    main()
