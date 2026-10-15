import pytest

from attacker.reconstruction_metrics import (
    automatic_span_annotations, reconstruction_metrics, validate_span_annotations,
)


def test_aligned_metrics_count_windows_and_spans():
    annotations = [dict(kind='name', start=0, end=2),
                   dict(kind='number', start=2, end=4),
                   dict(kind='identifier', start=4, end=8)]
    result = reconstruction_metrics(list(range(8)), [0, 1, 99, 3, 4, 5, 6, 7], annotations)
    assert result['correct_tokens'] == 7
    assert result['token_accuracy'] == 7 / 8
    assert result['sequence_exact'] is False
    assert result['longest_exact_run'] == 5
    assert result['contiguous_spans']['2'] == dict(exact=5, total=7, rate=5 / 7)
    assert result['contiguous_spans']['4'] == dict(exact=2, total=5, rate=2 / 5)
    assert result['contiguous_spans']['8']['exact'] == 0
    assert result['annotated_spans']['name']['rate'] == 1
    assert result['annotated_spans']['number']['rate'] == 0
    assert result['annotated_spans']['identifier']['rate'] == 1


def test_absent_spans_are_missing_and_all_wrong_has_zero_run():
    result = reconstruction_metrics([1], [2])
    assert result['longest_exact_run'] == 0
    assert result['contiguous_spans']['2'] == dict(exact=0, total=0, rate=None)
    assert result['annotated_spans']['name']['rate'] is None
    assert reconstruction_metrics([1], [1])['sequence_exact'] is True


def test_matching_substring_at_wrong_position_is_not_recovery():
    result = reconstruction_metrics([1, 2, 3, 4], [3, 4, 1, 2])
    assert result['correct_tokens'] == result['contiguous_spans']['2']['exact'] == 0


@pytest.mark.parametrize('truth,pred', [([], []), ([True], [1]), ([1], [-1]),
                                      ([1], [1, 2]), ('1', [1]), ([1.0], [1])])
def test_invalid_tokens_fail_closed(truth, pred):
    with pytest.raises(ValueError):
        reconstruction_metrics(truth, pred)


@pytest.mark.parametrize('annotation', [dict(kind='person', start=0, end=1),
    dict(kind='name', start=-1, end=1), dict(kind='name', start=0, end=3),
    dict(kind='name', start=1, end=1), dict(kind='name', start=True, end=2),
    dict(kind='name', start=0, end=1, label='x')])
def test_bad_annotations_fail(annotation):
    with pytest.raises(ValueError):
        validate_span_annotations([annotation], 2)


def test_duplicate_annotations_rejected_but_overlapping_categories_allowed():
    annotation = dict(kind='name', start=0, end=1)
    with pytest.raises(ValueError):
        validate_span_annotations([annotation, annotation], 2)
    assert len(validate_span_annotations([annotation, dict(kind='identifier', start=0, end=1)], 2)) == 2


def test_automatic_annotations_use_covering_tokens_not_names():
    text = 'Alice ID_42 paid 12.50 euros'
    offsets = [(0, 5), (5, 8), (8, 11), (11, 16), (16, 19), (19, 22), (22, 28)]
    assert automatic_span_annotations(text, offsets) == [
        dict(kind='identifier', start=1, end=3), dict(kind='number', start=4, end=6)]


def test_partial_spans_and_uncovered_character_gaps_are_omitted():
    assert automatic_span_annotations('ID_1234', [(0, 4)]) == []
    assert automatic_span_annotations('1234', [(0, 1), (2, 4)]) == []
    assert automatic_span_annotations('1234', [(1, 4)]) == []


def test_unicode_offsets_overlapping_bpe_and_special_tokens():
    assert automatic_span_annotations('é 42', [(0, 0), (0, 1), (0, 1), (1, 4), (0, 0)]) == [
        dict(kind='number', start=3, end=4)]


@pytest.mark.parametrize('offsets', [[(0, 8)], [(2, 3), (0, 2)], [(True, 2)], [(1, 1)]])
def test_invalid_offsets_rejected(offsets):
    with pytest.raises(ValueError):
        automatic_span_annotations('123', offsets)


@pytest.mark.parametrize('lengths', [[], [0], [True], [2, 2], '2'])
def test_invalid_window_lengths_rejected(lengths):
    with pytest.raises(ValueError):
        reconstruction_metrics([1, 2], [1, 2], span_lengths=lengths)


def test_hyphenated_ordinary_word_is_not_an_identifier():
    text = 'well-known CVE-123'
    assert automatic_span_annotations(text, [(0, 10), (10, 18)]) == [
        dict(kind='identifier', start=1, end=2)]
