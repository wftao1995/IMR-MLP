#!/usr/bin/env python3
"""Regression checks for documentation validation (standard library only)."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_doc_commands
import check_doc_consistency
import check_doc_locale
import doc_source_scan
from doxygen_markdown import LINK


class CommandChecks(unittest.TestCase):
    def test_paired_options_in_first_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'commands.md'
            path.write_text('### `run` (alias: `r`)\n\n'
                            '| `--source`, `--target` | Uses `--settings` when needed |\n')
            documented = doc_source_scan.documented_commands(path)['run']
        self.assertEqual(documented['options'], {'source', 'target'})
        self.assertEqual(documented['aliases'], {'r'})

    def run_check(self, document, strict=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'commands.md'
            path.write_text(document)
            argv = ['check_doc_commands'] + (['--strict'] if strict else [])
            with (patch.object(doc_source_scan, 'ROOT', root),
                  patch.object(check_doc_commands, 'DOCUMENTS', {'en': path}),
                  patch.object(check_doc_commands, 'ALLOWLIST', root / 'allowlist'),
                  patch.object(check_doc_commands, 'load_allowlist', return_value=set()),
                  patch.object(doc_source_scan, 'command_options', return_value={'run': {'input'}}),
                  patch.object(doc_source_scan, 'registrations',
                               return_value=({'run_impl': 'run'}, [('r', 'run_impl')])),
                  patch('sys.argv', argv), redirect_stdout(StringIO())):
                return check_doc_commands.main()

    def test_strict_accepts_matching_reference(self):
        self.assertEqual(self.run_check('### `run` (alias: `r`)\n| `--input` | File |\n'), 0)

    def test_strict_rejects_stale_option_and_alias(self):
        for document in ('### `run` (alias: `r`)\n| `--input` | File |\n| `--removed` | Old |\n',
                         '### `run`\n| `--input` | File |\n'):
            with self.subTest(document=document):
                self.assertEqual(self.run_check(document), 1)

    def test_strict_rejects_missing_option_or_command(self):
        for document in ('### `run` (alias: `r`)\n', '### `other`\n'):
            with self.subTest(document=document):
                self.assertEqual(self.run_check(document), 1)


class ConsistencyChecks(unittest.TestCase):
    def run_check(self, en_document, zh_document):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            en_path, zh_path = root / 'commands.md', root / 'commands_zh.md'
            en_path.write_text(en_document)
            zh_path.write_text(zh_document)
            with (patch.object(check_doc_consistency, 'DOC_EN', en_path),
                  patch.object(check_doc_consistency, 'DOC_ZH', zh_path),
                  redirect_stdout(StringIO())):
                return check_doc_consistency.main()

    EN = ('### `run` — Run\n\n'
          '| Option | Alias | Default | Description |\n'
          '|--------|-------|---------|-------------|\n'
          '| `--input` | `-in` | `POSCAR` | Input file |\n')
    ZH = ('### `run` — \u8fd0\u884c\n\n'
          '| \u9009\u9879 | \u522b\u540d | \u9ed8\u8ba4\u503c | \u63cf\u8ff0 |\n'
          '|--------|-------|---------|-------------|\n'
          '| `--input` | `-in` | `POSCAR` | \u8f93\u5165\u6587\u4ef6 |\n')

    def test_matching_tables_pass(self):
        self.assertEqual(self.run_check(self.EN, self.ZH), 0)

    def test_dropped_default_and_alias_columns_fail(self):
        narrow = ('### `run` — \u8fd0\u884c\n\n'
                  '| \u9009\u9879 | \u63cf\u8ff0 |\n'
                  '|--------|-------------|\n'
                  '| `--input` | \u8f93\u5165\u6587\u4ef6 |\n')
        self.assertEqual(self.run_check(self.EN, narrow), 1)

    def test_missing_option_still_fails(self):
        zh = self.ZH.replace('| `--input` | `-in` | `POSCAR` | \u8f93\u5165\u6587\u4ef6 |\n', '')
        self.assertEqual(self.run_check(self.EN, zh), 1)


class LocaleChecks(unittest.TestCase):
    def test_multiline_label_is_a_link(self):
        match = LINK.search('[Single-channel energy and\nself terms](../reference/sog.md)')
        self.assertIsNotNone(match)
        self.assertEqual(match[2], '../reference/sog.md')

    def test_checks_examples_outside_manual_and_skips_build_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'examples/build').mkdir(parents=True)
            entry = root / 'examples/README.md'
            entry.write_text('[中文](README_zh.md)\n[Guide](guide_zh.md)\n'
                             '```text\n[Example](missing_zh.md)\n```\n')
            (root / 'examples/build/generated.md').write_text('# Generated\n')
            with patch.object(check_doc_locale, 'ROOT', root):
                self.assertEqual(check_doc_locale.markdown_sources(), [entry])
                self.assertEqual(list(check_doc_locale.relative_links(entry)),
                                 [('guide_zh.md', root / 'examples/guide_zh.md')])


if __name__ == '__main__':
    unittest.main()
