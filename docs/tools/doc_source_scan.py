#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Shared parsers that map C++ sources to the documented CLI surface.

The scanners are intentionally small and dependency-free: they read the
command registry in ``apps/apps.cpp``, the option registrations inside each
command function, the shared MaxVol console options, and the Markdown command
sections.  Other documentation checks import this module so that the expected
command/option sets are derived from the code in exactly one place.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPS = ROOT / "apps"
MASKS = ("//", "/*")

REG_COMMAND_RE = re.compile(
    r"^\s*REG_COMMAND\s*\(\s*([A-Za-z_]\w*)\s*,", re.MULTILINE)
REG_NICK_RE = re.compile(
    r'REG_NICK(?:NAME|CMD)\s*\(\s*"([^"]+)"\s*,\s*([A-Za-z_]\w*)', re.MULTILINE)
ADDOP_RE = re.compile(
    r'\baddop\s*\(\s*[^,]+,\s*"([A-Za-z0-9_:.\-]+)"')
CONSOLE_RE = re.compile(
    r'\bConnectWithConsole\s*\(\s*[^,]+,\s*[^,]+,\s*"([A-Za-z0-9_:.\-]+)"')
FUNCTION_RE = re.compile(
    r"^(?:static\s+)?int\s+([A-Za-z_]\w*)\s*\(\s*int\s+argc\b", re.MULTILINE)


def _files():
    return sorted(APPS.glob("*.cpp"))


def registrations():
    """Return ``(canonical, aliases)`` parsed from ``apps/apps.cpp``.

    ``canonical`` maps the implementing function name to the primary command
    name; ``aliases`` is a list of ``(alias, function)`` pairs.  A function
    registered through ``REG_NICKCMD``/``REG_NICKNAME`` only is promoted to a
    canonical command under that nickname.
    """
    text = (APPS / "apps.cpp").read_text(encoding="utf-8")
    canonical = {name: name for name in REG_COMMAND_RE.findall(text)}
    aliases = REG_NICK_RE.findall(text)
    for alias, function in aliases:
        canonical.setdefault(function, alias)
    return canonical, aliases


def command_sources():
    """Map every canonical command name to the file defining its entry point."""
    canonical, _ = registrations()
    definitions = {}
    for path in _files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for function in FUNCTION_RE.findall(text):
            definitions.setdefault(function, path)
    result = {}
    for function, name in canonical.items():
        if function in definitions:
            result[name] = definitions[function]
    return result


def function_body(path: Path, function: str) -> str:
    """Return the source text of ``int function(int argc, ...)``."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"^(?:static\s+)?int\s+" + re.escape(function) +
        r"\s*\(\s*int\s+argc\b", text, re.MULTILINE)
    if not match:
        return ""
    start = match.start()
    following = FUNCTION_RE.search(text, match.end())
    return text[start:following.start() if following else len(text)]


def console_option_groups():
    """Return ``(builder_options, selector_options)`` shared by MaxVol apps."""
    builder_options = set()
    builder = APPS.parent / "maxvol" / "b_matrix_builder.cpp"
    if builder.is_file():
        text = builder.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r"RegisterConsole.*?for\s*\([^)]*?\{([^}]*)\}\s*\)", text, re.DOTALL)
        if match:
            builder_options.update(re.findall(r'"([^"]+)"', match.group(1)))
    selector_options = set()
    selector = APPS.parent / "maxvol" / "active_selector.cpp"
    if selector.is_file():
        text = selector.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r"void\s+ActiveSelector::RegisterConsole.*?(?=\n\}|^[A-Za-z].*::)",
            text, re.DOTALL | re.MULTILINE)
        if match:
            selector_options.update(ADDOP_RE.findall(match.group(0)))
    return builder_options, selector_options


def console_options():
    """Options registered by the shared MaxVol builder/selector consoles."""
    builder_options, selector_options = console_option_groups()
    return builder_options | selector_options


def strip_comments(text: str) -> str:
    """Remove C/C++ comments so commented-out options are not registered."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def command_options():
    """Return ``{command: set(long option names)}`` derived from the sources.

    Options are often registered by a settings class rather than inside the
    ``int cmd(int argc, ...)`` body.  For files that implement exactly one
    canonical command, scan the whole file; for shared files (for example
    ``cellutils.cpp``) keep the per-function scope to avoid cross-assignment.
    """
    sources = command_sources()
    canonical, _ = registrations()
    reverse = {}
    for function, name in canonical.items():
        reverse.setdefault(name, function)
    per_file = {}
    for name, path in sources.items():
        per_file[path] = per_file.get(path, 0) + 1
    builder_options, selector_options = console_option_groups()
    result = {}
    for name, path in sources.items():
        text = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
        if per_file[path] == 1:
            scope = text
        else:
            scope = strip_comments(function_body(path, reverse[name]))
        options = (set(ADDOP_RE.findall(scope)) |
                   set(CONSOLE_RE.findall(scope)))
        if "builder_settings.RegisterConsole" in scope:
            options |= builder_options
        if "selector_settings.RegisterConsole" in scope:
            options |= selector_options
        options = {option for option in options if not option.endswith(":")}
        result[name] = options
    return result


def documented_commands(path: Path):
    """Return ``{command: {"options": set(), "heading": str, "aliases": set()}}``."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections = re.split(r"(?=^### )", text, flags=re.MULTILINE)
    result = {}
    for section in sections:
        match = re.match(r"^### `([^`]+)`[^\n]*", section)
        if not match:
            continue
        name = match.group(1)
        heading = match.group(0)
        # A first cell may list a pair such as --source-map, --target-map.
        # Scan all names in that cell, excluding options mentioned in descriptions.
        options = set()
        for cell in re.findall(r"^\|([^|\n]*)\|", section, re.MULTILINE):
            options.update(re.findall(r"`--([A-Za-z0-9_:.\-]+)`", cell))
        aliases = set(re.findall(
            r"(?:alias(?:es)?|别名)\s*[:：]\s*([^\n]*)", heading))
        alias_names = set(re.findall(r"`([^`]+)`", " ".join(aliases)))
        result[name] = {"options": options, "heading": heading,
                        "aliases": alias_names}
    return result
