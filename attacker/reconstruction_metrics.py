"""Truth-side, position-aligned reconstruction metrics; never attack inputs.

Span annotations use half-open token intervals. A span succeeds only when every
covering token is exact, including any surrounding characters in boundary tokens.
Automatic annotations are lexical proxies, not sensitive-data or NER labels.
"""
from collections.abc import Mapping, Sequence
import re

SPAN_KINDS = ('name', 'number', 'identifier')
_IDENTIFIER = re.compile(r'(?<!\w)[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?!\w)')
_NUMBER = re.compile(r'(?<![\w.])[-+]?\d+(?:[,.]\d+)*(?:%|\b)(?!\w)')


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _tokens(value, label):
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or not value or any(not _integer(v) or v < 0 for v in value)):
        raise ValueError(f'{label} must be a nonempty sequence of nonnegative integer token IDs')
    return list(value)


def validate_span_annotations(annotations, token_count):
    """Return canonical annotations, rejecting malformed or duplicate intervals."""
    if not _integer(token_count) or token_count <= 0:
        raise ValueError('token_count must be a positive integer')
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
        raise ValueError('annotations must be a sequence')
    result, seen = [], set()
    for item in annotations:
        if not isinstance(item, Mapping) or set(item) != {'kind', 'start', 'end'}:
            raise ValueError('annotations require exactly kind, start and end')
        kind, start, end = item['kind'], item['start'], item['end']
        if (not isinstance(kind, str) or kind not in SPAN_KINDS
                or not _integer(start) or not _integer(end)
                or not 0 <= start < end <= token_count):
            raise ValueError('invalid annotation kind or token interval')
        key = (kind, start, end)
        if key in seen:
            raise ValueError('duplicate span annotation')
        seen.add(key)
        result.append(dict(kind=kind, start=start, end=end))
    return sorted(result, key=lambda a: (a['start'], a['end'], a['kind']))


def reconstruction_metrics(truth, prediction, annotations=(), span_lengths=(2, 4, 8)):
    """Score a single document; missing/extra predictions fail closed.

    Contiguous-span rates count all overlapping windows at their original token
    positions, not substring presence elsewhere. No applicable spans yields None,
    never zero or a perfect score. Keep integer counts for pooled aggregation and
    retain document rows for uncertainty estimation; tokens are not independent.
    """
    truth, prediction = _tokens(truth, 'truth'), _tokens(prediction, 'prediction')
    if len(truth) != len(prediction):
        raise ValueError('truth and prediction lengths must match')
    if (not isinstance(span_lengths, Sequence) or isinstance(span_lengths, (str, bytes))
            or not span_lengths or any(not _integer(k) or k <= 0 for k in span_lengths)
            or len(set(span_lengths)) != len(span_lengths)):
        raise ValueError('span_lengths must contain unique positive integers')
    annotations = validate_span_annotations(annotations, len(truth))
    matches = [a == b for a, b in zip(truth, prediction)]
    longest, run = 0, 0
    for match in matches:
        run = run + 1 if match else 0
        longest = max(longest, run)
    windows = {}
    for length in span_lengths:
        total = max(0, len(truth) - length + 1)
        exact = sum(all(matches[i:i + length]) for i in range(total))
        windows[str(length)] = {'exact': exact, 'total': total,
                                'rate': exact / total if total else None}
    spans = {}
    for kind in SPAN_KINDS:
        selected = [a for a in annotations if a['kind'] == kind]
        total = len(selected)
        exact = sum(all(matches[a['start']:a['end']]) for a in selected)
        spans[kind] = {'exact': exact, 'total': total,
                       'rate': exact / total if total else None}
    return {'token_count': len(truth), 'correct_tokens': sum(matches),
            'token_accuracy': sum(matches) / len(truth),
            'sequence_exact': all(matches), 'longest_exact_run': longest,
            'contiguous_spans': windows, 'annotated_spans': spans}


def automatic_span_annotations(text, offsets):
    """Find numbers and ASCII identifier proxies with frozen regex rules.

    Identifiers begin with a letter and contain a digit or underscore (hyphens
    are permitted internally). Hyphenated ordinary words alone do not qualify.
    Number matches inside identifiers are excluded. Only complete matches covered
    by supplied character offsets are scored: truncated suffixes are omitted.
    Offsets must index the exact original text, including Unicode characters.
    Special (0, 0) offsets are allowed; byte-BPE overlapping offsets are supported.
    Caller must retain text/offset provenance and add human name annotations if
    name recovery is to be evaluated. Duplicate token intervals are counted once.
    """
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)) or not offsets:
        raise ValueError('offsets must be a nonempty sequence')
    clean, previous = [], (0, 0)
    for pair in offsets:
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2
                or any(not _integer(v) for v in pair)):
            raise ValueError('offsets require integer start/end pairs')
        start, end = pair
        if not 0 <= start <= end <= len(text) or (start == end and (start, end) != (0, 0)):
            raise ValueError('invalid character offsets')
        if start != end:
            if start < previous[0] or end < previous[1]:
                raise ValueError('offsets must follow text order')
            previous = (start, end)
        clean.append((start, end))
    identifiers = [m for m in _IDENTIFIER.finditer(text)
                   if any(c.isdigit() or c == '_' for c in m.group())]
    numbers = [m for m in _NUMBER.finditer(text)
               if not any(m.start() < i.end() and i.start() < m.end() for i in identifiers)]
    found = set()
    for kind, matches in [('identifier', identifiers), ('number', numbers)]:
        for match in matches:
            covered = [i for i, (start, end) in enumerate(clean)
                       if start < match.end() and end > match.start()]
            if not covered:
                continue
            cursor = match.start()
            for i in covered:
                start, end = clean[i]
                if start > cursor:
                    break
                cursor = max(cursor, end)
            if cursor < match.end():
                continue
            if any(clean[i][0] == clean[i][1] for i in range(covered[0], covered[-1] + 1)):
                raise ValueError('special token interrupts a lexical span')
            found.add((kind, covered[0], covered[-1] + 1))
    return validate_span_annotations([dict(kind=k, start=s, end=e) for k, s, e in found], len(clean))
