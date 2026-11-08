"""Train-only extraction of FlowER radical elementary steps (RC/RS).

These labels are dataset classes, not pathway IDs. Never group rows into a
mechanism or fabricate arrows from the reactant/product pair.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase


_NON_TRAIN_SEGMENTS = {'test', 'tests', 'val', 'valid', 'validation'}


def _is_non_train_segment(segment: str) -> bool:
    text = segment.lower()
    return text in _NON_TRAIN_SEGMENTS or 'holdout' in text


def _check_train_only(given: Path, resolved: Path) -> None:
    if resolved.name != 'train.txt' or given.name != 'train.txt':
        raise ValueError('Only a FlowER train.txt source is permitted')
    for path in (given, resolved):
        if any(_is_non_train_segment(part) for part in path.parts[:-1]):
            raise ValueError(
                'Only a FlowER train.txt source is permitted; refusing a path inside a '
                'test/validation/holdout directory'
            )


def extract_radical_training_cases(
    source: Path, *, limit: int | None = None, per_label: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract unique RC/RS rows from a FlowER ``train.txt``.

    ``per_label`` keeps the first N cases (by content-derived ID) of each
    label; ``limit`` then keeps the first N overall. The report records the
    source path as given, so a repo-relative invocation stays reproducible.
    """
    given = Path(source)
    source = given.resolve()
    _check_train_only(given, source)
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if per_label is not None and per_label < 1:
        raise ValueError('per_label must be positive')
    digest = hashlib.sha256()
    counts: Counter = Counter()
    records: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    params = Chem.SmilesParserParams()
    params.removeHs = False
    with source.open('rb') as handle, rdBase.BlockLogs():
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            reaction, _, label = raw.decode('utf-8').strip().rpartition('|')
            if label not in {'RC', 'RS'}:
                continue
            counts['radical_rows'] += 1
            counts[label] += 1
            sides = reaction.split('>>')
            mols = [Chem.MolFromSmiles(s, params) for s in sides] if len(sides) == 2 else []
            error = None
            if len(mols) != 2 or any(m is None for m in mols):
                error = 'invalid_reaction'
            elif not any(a.GetNumRadicalElectrons() for m in mols for a in m.GetAtoms()):
                error = 'no_radical_electrons'
            if error:
                counts['rejected_rows'] += 1
                rejected.append({'line': line_number, 'label': label, 'reason': error})
                continue
            row_hash = hashlib.sha256(f'{label}|{reaction}'.encode()).hexdigest()
            if row_hash in records:
                counts['duplicate_rows'] += 1
                records[row_hash]['provenance']['source_lines'].append(line_number)
                continue
            records[row_hash] = {
                'id': f'flower_radical_{label.lower()}_{row_hash[:20]}',
                'name': f'FlowER {label} elementary radical step',
                'starting_materials': sides[0].split('.'),
                'products': sides[1].split('.'),
                'tags': ['radical', 'elementary_step', 'train_derived'],
                'notes': 'Development step case; no curated arrow annotations or full pathway. Not a held-out quality claim.',
                'reference_reaction_smirks': reaction,
                'provenance': {
                    'source_dataset': 'FlowER flower_new_dataset',
                    'source_split': 'train', 'source_label': label,
                    'source_lines': [line_number], 'record_sha256': row_hash,
                    'source_url': 'https://figshare.com/articles/dataset/28359407/3',
                },
            }
    cases = sorted(records.values(), key=lambda case: case['id'])
    exported = cases
    if per_label is not None:
        taken: Counter = Counter()
        exported = []
        for case in cases:
            label = case['provenance']['source_label']
            if taken[label] < per_label:
                taken[label] += 1
                exported.append(case)
    if limit is not None:
        exported = exported[:limit]
    report = {
        'schema_version': 'radical_training_extract.v1',
        'source': given.as_posix(), 'source_split': 'train',
        'source_sha256': digest.hexdigest(),
        **{k: counts[k] for k in ('radical_rows', 'duplicate_rows', 'rejected_rows')},
        'labels': {k: counts[k] for k in ('RC', 'RS')},
        'unique_cases': len(cases), 'exported_cases': len(exported),
        'per_label': per_label, 'limit': limit,
        'rejections': rejected,
    }
    for case in exported:
        case['provenance']['source_sha256'] = report['source_sha256']
    return exported, report
