#!/usr/bin/env python3
"""Inventory real MTP contraction graphs and verify the portable fixture copies."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def integer(text, name):
    match = re.search(r'^\s*' + re.escape(name) + r'\s*=\s*(\d+)', text, re.M)
    if not match:
        raise ValueError(f'missing integer field {name}')
    return int(match[1])


def indices(text, name, width=1):
    match = re.search(r'^\s*' + re.escape(name) + r'\s*=\s*\{', text, re.M)
    if not match:
        raise ValueError(f'missing array {name}')
    begin = match.end() - 1
    depth = 0
    for end in range(begin, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            break
    else:
        raise ValueError(f'unterminated array {name}')
    raw = text[begin : end + 1]
    if re.sub(r'[\s{},+\-\d]', '', raw):
        raise ValueError(f'invalid integer array {name}')
    values = list(map(int, re.findall(r'[+-]?\d+', raw)))
    if len(values) % width:
        raise ValueError(f'invalid dimensions for {name}')
    return (
        [tuple(values[i : i + width]) for i in range(0, len(values), width)]
        if width > 1
        else values
    )


def inspect(path, root):
    raw = path.read_bytes()
    text = raw.decode()
    basic = indices(text, 'alpha_index_basic', 4)
    rules = indices(text, 'alpha_index_times', 4)
    mapping = indices(text, 'alpha_moment_mapping')
    n = integer(text, 'alpha_moments_count')
    nb = integer(text, 'alpha_index_basic_count')
    if (len(basic), len(rules), len(mapping)) != (
        nb,
        integer(text, 'alpha_index_times_count'),
        integer(text, 'alpha_scalar_moments'),
    ):
        raise ValueError(f'{path}: declared counts differ from stored arrays')
    if not (0 < nb <= n) or not mapping:
        raise ValueError(f'{path}: invalid moment/scalar count')
    levels, defined = [0] * n, set(range(nb))
    for a, b, coefficient, target in rules:
        if not (
            a in defined
            and b in defined
            and nb <= target < n
            and 0 <= a < target
            and 0 <= b < target
        ):
            raise ValueError(f'{path}: invalid rule {(a, b, coefficient, target)}')
        levels[target] = max(levels[target], levels[a] + 1, levels[b] + 1)
        defined.add(target)
    if any(i not in defined for i in mapping) or len(set(mapping)) != len(mapping):
        raise ValueError(f'{path}: undefined or duplicate scalar mapping')
    order = sorted(range(nb), key=lambda i: (sum(basic[i][1:]), basic[i][0]))
    basic_ids = dict(zip(order, range(nb)))
    selected = set(mapping)
    grouped = sorted(
        rules,
        key=lambda r: (
            levels[r[3]],
            basic_ids.get(r[0], r[0]),
            basic_ids.get(r[1], r[1]),
            -int(r[3] in selected),
        ),
    )
    ids, runs, previous = {}, Counter(), None
    for a, b, coefficient, target in grouped:
        if target not in ids:
            ids[target] = nb + len(ids)
        if target != previous:
            runs[target] += 1
            previous = target
    mapped = sorted(basic_ids[i] if i < nb else ids[i] for i in mapping)
    additions = Counter(r[3] for r in rules)
    return dict(
        path=path.relative_to(root).as_posix(),
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        moments=n,
        basic=nb,
        products=len(rules),
        scalars=len(mapping),
        depth=max(levels),
        max_rank=max(sum(b[1:]) for b in basic),
        max_accumulation=max(additions.values(), default=0),
        interleaved_targets=sum(count > 1 for count in runs.values()),
        scalar_gaps_after_group=sum(b != a + 1 for a, b in zip(mapped, mapped[1:])),
        invalid_rules=0,
        coefficients=bool(re.search(r'^\s*radial_coeffs\b', text, re.M)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='MTM-correct directory (read only)')
    parser.add_argument('--output', type=Path, help='Write inventory JSON')
    parser.add_argument(
        '--fixture-dir',
        type=Path,
        default=Path(__file__).resolve().parents[2] / 'unitest/test-data/mtp-basis-corpus',
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f'corpus directory does not exist: {root}')
    rows = [inspect(p, root) for p in sorted(root.rglob('potential.mtp_0'))]
    if not rows:
        parser.error('no potential.mtp_0 files found')
    manifest = json.loads((args.fixture_dir / 'sources.json').read_text())
    expected = {row['path']: row for row in manifest['files']}
    by_path = {row['path']: row for row in rows}
    if set(by_path) != set(expected):
        raise ValueError('corpus file set differs from sources.json')
    copied = 0
    for row in rows:
        old = expected[row['path']]
        if row['sha256'] != old['sha256']:
            raise ValueError(f"source checksum changed: {row['path']}")
        if old['fixture']:
            fixture = args.fixture_dir / old['fixture']
            if hashlib.sha256(fixture.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'fixture checksum changed: {fixture}')
            if row['coefficients']:
                raise ValueError(f'fixture unexpectedly contains fitted coefficients: {fixture}')
            copied += 1
        print(
            f"{row['path']}: basic={row['basic']} moments={row['moments']} products={row['products']} scalars={row['scalars']} depth={row['depth']} rank={row['max_rank']} interleaved={row['interleaved_targets']}"
        )
    report = dict(source='MTM-correct', files=rows, verified_fixtures=copied)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Validated {len(rows)} source graphs and {copied} portable fixtures')


if __name__ == '__main__':
    main()
