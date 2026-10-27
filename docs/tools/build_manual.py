#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Build the categorized manual from canonical Markdown and the shared catalog."""
import argparse
from html import escape
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from urllib.parse import unquote, urlsplit

from doxygen_markdown import ROOT, LINK, HEADING, local_target, page_id, preferred_page, prose_parts, sections
from doc_catalog import categories, entries, localized, title
from check_doc_html import check_html
from build_docs import copy_mathjax
from check_doc_math import check_math
from publish_docs import publish_directory


def manual_pages():
    return [dict(c, slug=c['id'] + '/index', render=True) for c in categories()] + [p for p in entries() if p['render']]


def language_switcher(language, page):
    labels = {'zh': '中文', 'en': 'English'}
    caption = '切换语言' if language == 'zh' else 'Switch language'
    links = []
    for target_language, label in labels.items():
        target = posixpath.relpath(
            str(PurePosixPath(target_language, page)),
            str(PurePosixPath(language, page).parent))
        current = ' aria-current="page"' if target_language == language else ''
        links.append(f'<a href="{escape(target, quote=True)}" lang="{target_language}" '
                     f'hreflang="{target_language}"{current}>{label}</a>')
    return ('<details class="imr-language-switch">'
            f'<summary aria-label="{caption}: {labels[language]}">{labels[language]}</summary>'
            f'<nav aria-label="{caption}">{"".join(links)}</nav></details>')


def display_math(match):
    # A display equation inside a list must retain the list's indentation.
    # Otherwise MyST ends the list and reads the next numbered step as prose.
    prefix = match.string[match.string.rfind('\n', 0, match.start()) + 1:match.start()]
    indent = prefix if not prefix.strip() else ''
    body = textwrap.dedent(match[1]).strip()
    return '\n' + indent + '$$\n' + textwrap.indent(body, indent) + '\n' + indent + '$$\n'


def convert(source, language, slug=None):
    source = preferred_page(source, language)
    pages = manual_pages()
    by_source = {localized(ROOT / p['source'], language): p for p in pages}
    slug = slug or by_source[source]['slug']
    current = PurePosixPath('manual-html', language, slug + '.html')

    def relative(target):
        return posixpath.relpath(str(target), str(current.parent))

    def doc_url(destination, requested_language, fragment=''):
        destination = preferred_page(destination, requested_language)
        targets = {localized(ROOT / p['source'], requested_language): p['slug'] for p in pages}
        if destination in targets:
            target = relative(PurePosixPath('manual-html', requested_language, targets[destination] + '.html'))
        else:
            identifier = page_id(destination)
            target = relative(PurePosixPath('build_zh' if requested_language == 'zh' else 'build',
                                           'html', ('index' if identifier in ('mainpage', 'index') else identifier) + '.html'))
        if fragment:
            anchor = sections(destination).get(unquote(fragment))
            if not anchor:
                raise ValueError(f'{source}: unknown section {destination}#{fragment}')
            target += '#' + anchor
        return target

    def url(target, label=''):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:return target
        if parsed.path.endswith('.html'):
            # Source documents use these only for generated C++ indexes.
            if parsed.path in ('annotated.html','files.html','topics.html','todo.html','bug.html','deprecated.html'):
                return relative(PurePosixPath('build_zh' if language == 'zh' else 'build','html',parsed.path))
            if 'manual-html/' in parsed.path:
                return relative(PurePosixPath('manual-html', parsed.path.split('manual-html/',1)[1]))
            return target
        destination = local_target(source, target)
        if destination is None:return None
        requested_language = {'English':'en','中文':'zh','简体中文':'zh'}.get(label, language)
        if destination.is_dir():destination /= 'README.md'
        if destination.suffix == '.md':return doc_url(destination, requested_language, parsed.fragment)
        return relative(PurePosixPath('build_zh' if language == 'zh' else 'build', 'html/assets',destination.relative_to(ROOT).as_posix()))

    def link(match):
        label = match[1].removeprefix('[').removesuffix('](')
        target = url(match[2],label)
        return match[1]+target+match[3] if target else label+' (`'+match[2]+'`)'

    all_sources = [localized(ROOT/p['source'],language) for p in [*categories(),*entries()]]
    refs = {page_id(p):p for p in all_sources}
    groups = set(re.findall(r'@defgroup\s+(\w+)',(ROOT/'docs/groups.dox').read_text()))
    def ref_link(identifier, label=None):
        label = label or (title(refs[identifier]) if identifier in refs else identifier)
        if identifier in refs:target = doc_url(refs[identifier],language)
        elif identifier == 'module_index':
            target=relative(PurePosixPath('build_zh' if language=='zh' else 'build','html/index.html'))+'#module_index'
        elif identifier in groups:
            target=relative(PurePosixPath('build_zh' if language=='zh' else 'build','html/group__'+identifier.replace('_','__')+'.html'))
        else:raise ValueError(f'{source}: unknown Doxygen reference {identifier}')
        return f'[{label}]({target})'

    result=[]
    heading_ids=sections(source)
    from doxygen_markdown import heading_slug
    counts={}
    def heading(match):
        key=heading_slug(match[2]); count=counts.get(key,0);counts[key]=count+1
        key += f'-{count}' if count else ''
        text=re.sub(r'\s*\{#[^}]+\}', '',match[0])
        # Docutils normalizes underscores in explicit labels. Preserve literal
        # Doxygen anchors too so cross-format and old bookmarked URLs work.
        return '<a id="'+heading_ids[key]+'"></a>\n\n'+text
    for i,part in enumerate(prose_parts(source.read_text())):
        if i%2:
            result.append(part);continue
        part=part.replace('@tableofcontents','')
        # Source Markdown keeps its language links; HTML uses the toolbar menu.
        part=re.sub(r'(?m)^\[(?:English|中文|简体中文)\]\([^\n)]+\)[ \t]*(?:[·|][ \t]*)?', '', part)
        part=HEADING.sub(heading,part)
        # Handle link-wrapped Doxygen references before ordinary links.
        part=re.sub(r'\[([^\]]+)\]\((?:@|\\)ref\s+(\w+)\)',lambda m:ref_link(m[2],m[1]),part)
        part=LINK.sub(link,part)
        part=re.sub(r'(?m)^(\[[^\]\n]+\]:\s*)(\S+)',lambda m:m[1]+(url(m[2]) or m[2]),part)
        part=re.sub(r'(?:@ref|\\(?:ref|subpage))\s+(\w+)(?:\s+"([^"\n]+)")?',lambda m:ref_link(m[1],m[2]),part)
        # Math remains local through Sphinx's MathJax extension.
        part=re.sub(r'(?:@|\\)f\$(.*?)(?:@|\\)f\$',lambda m:'$'+m[1]+'$',part,flags=re.S)
        part=re.sub(r'(?:@|\\)f\[(.*?)(?:@|\\)f\]',display_math,part,flags=re.S)
        part=re.sub(r'\\\((.*?)\\\)',lambda m:'$'+m[1]+'$',part,flags=re.S)
        part=re.sub(r'\\\[(.*?)\\\]',display_math,part,flags=re.S)
        part=re.sub(r'(?m)^@(?:note|warning)\s+', '> ',part)
        result.append(part)
    text=''.join(result)
    notices = []
    # Clearly label an original when no counterpart exists in the requested language.
    translated = source.stem.endswith('_zh') if language=='zh' else not source.stem.endswith('_zh')
    if not translated:
        notice='此页暂无中文译文，以下为英文原文。' if language=='zh' else 'This page has no English translation; the Chinese original follows.'
        notices.append('> ' + notice)
    if notices:
        text = re.sub(r'(?m)^# .+$', lambda m: m[0] + '\n\n' + '\n\n'.join(notices), text, count=1)
    return text


def build_manual(output, python=sys.executable):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    (ROOT / 'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='manual-',dir=ROOT/'build') as temporary, \
            tempfile.TemporaryDirectory(prefix='.manual-',dir=output.parent) as staged:
        work,staging=Path(temporary),Path(staged)
        for language in ('en','zh'):
            source=work/language;source.mkdir()
            for page in manual_pages():
                path=source/(page['slug']+'.md');path.parent.mkdir(parents=True,exist_ok=True)
                content=convert(ROOT/page['source'],language,page['slug'])
                if 'groups' in page:
                    for group in page['groups']:
                        children=[p['slug'] for p in group['pages'] if p['render']]
                        if children:
                            content+='\n```{toctree}\n:hidden:\n:maxdepth: 1\n:caption: '+group['title'][language]+'\n\n'
                            content+='\n'.join('/'+child for child in children)+'\n```\n'
                path.write_text(content)
            shutil.copy2(ROOT/'docs/manual'/f'index_{language}.md',source/'index.md')
            static=source/'_static';static.mkdir()
            shutil.copy2(ROOT/'docs/IMR-MLP-logo.png',static/'IMR-MLP-logo.png')
            shutil.copy2(ROOT/'docs/manual/home.css',static/'home.css')
            shutil.copy2(ROOT/'docs/manual/language-switcher.js',static/'language-switcher.js')
            (source/'conf.py').write_text(
                "project='IMR-MLP'\nextensions=['myst_parser','sphinx.ext.mathjax']\nhtml_theme='furo'\n"
                f"language={'zh_CN' if language=='zh' else 'en'!r}\n"
                "myst_enable_extensions=['dollarmath','amsmath']\nmyst_heading_anchors=6\n"
                "myst_all_links_external=True\nhtml_title='IMR-MLP'\n"
                "html_static_path=['_static']\nhtml_css_files=['home.css']\n"
                "html_js_files=['language-switcher.js']\n"
                "html_logo='_static/IMR-MLP-logo.png'\n"
                "html_show_sourcelink=False\nhtml_show_sphinx=False\nhtml_show_copyright=False\n"
                # A local URL is made relative to each page by the builder below.
                "mathjax_path='mathjax-placeholder.js'\n"
                "html_theme_options={'navigation_with_keys':False}\n")
            with (source/'conf.py').open('a') as config:
                config.write("from pygments.lexers.special import TextLexer\n"
                             "def setup(app):\n    app.add_lexer('lammps', TextLexer)\n")
            subprocess.run([python,'-m','sphinx','-W','--keep-going','-b','html',str(source),str(staging/language)],check=True)
            copy_mathjax(staging / language)
            for html in (staging / language).rglob('*.html'):
                target = PurePosixPath('assets/mathjax/es5/tex-svg.js')
                current = PurePosixPath(html.relative_to(staging / language).as_posix())
                content = re.sub(r'(src=")([^" ]*mathjax-placeholder\.js)(")',
                                 lambda m: m[1] + posixpath.relpath(str(target), str(current.parent)) + m[3],
                                 html.read_text())
                switcher = language_switcher(language, current)
                for marker in ('<div class="content-icon-container">', '<div class="header-right">'):
                    if content.count(marker) != 1:
                        raise RuntimeError(f'{html}: missing or duplicated toolbar: {marker}')
                    content = content.replace(marker, marker + switcher, 1)
                html.write_text(content)
            redirects = {'install': 'guides/installation', 'quickstart': 'guides/quickstart',
                         'training': 'guides/workflows'}
            redirects.update({alias: page['slug'] for page in manual_pages()
                              for alias in page.get('aliases', [])})
            for old, new in redirects.items():
                path = staging / language / (old + '.html')
                path.parent.mkdir(parents=True, exist_ok=True)
                target = posixpath.relpath(new + '.html', str(PurePosixPath(old).parent))
                path.write_text('<!doctype html><meta charset="utf-8">'
                    f'<meta http-equiv="refresh" content="0;url={target}"><a href="{target}">Continue</a>'
                    f'<script>location.replace({json.dumps(target)}+location.search+location.hash)</script>')
        for language in ('en','zh'):
            errors,count=check_html(staging/language,output.parent)
            if errors:raise RuntimeError('\n'.join(errors))
            print(f'Manual {language}: {count} pages validated')
        subprocess.run(['node',str(ROOT/'docs/tools/check_doc_search.cjs'),str(staging/'en'),str(staging/'zh')],check=True)
        math = check_math([staging / 'en', staging / 'zh'])
        if math['errors']:
            raise RuntimeError('Manual formula errors: ' + json.dumps(math['errors'], ensure_ascii=False))
        print(f"Manual MathJax: {math['formulas']} formulas validated")
        publish_directory(staging,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/manual-html')
    args=parser.parse_args()
    build_manual(args.output)
