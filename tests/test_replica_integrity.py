import json

import pytest

from attacker.attacks import replica_integrity as ri
from attacker.evidence import verify_evidence


def test_contract_and_self_test():
    from attacker.attacks import validate_contract
    assert validate_contract(ri) is ri
    assert ri.self_test() == 0


def test_safety_availability_and_heterogeneity(tmp_path):
    path = tmp_path / 'battery.json'
    args = ri.build_parser().parse_args(['--toy', '--quick', '--output', str(path)])
    assert ri.run(args) == 0
    art = json.loads(path.read_text())
    summaries = {r['arm']: r for r in art['summary']}
    assert summaries['clean']['false_positive_rate'] == 0
    assert summaries['clean']['accepted_error_rate'] == 0
    assert summaries['one-malicious']['accepted_error_rate'] == 0
    assert summaries['two-colluding']['accepted_error_rate'] == 1
    assert summaries['two-colluding']['alarm_rate'] == 1
    assert summaries['k2-disagreement']['availability_abort_rate'] == 1
    assert summaries['k2-disagreement']['accepted_error_rate'] == 0
    assert summaries['honest-heterogeneity']['accepted_error_rate'] == 1
    assert summaries['honest-heterogeneity']['alarm_rate'] == 0
    assert all(r['compromised_replicas'] == 3 for r in art['results'] if r['arm'] == 'shared-key-disclosure')
    assert verify_evidence(str(path)+'.evidence.json')[0]
    with pytest.raises(ValueError, match='fresh'):
        ri.run(args)


def test_honest_majority_only_bounds_coordinates():
    result = ri.aggregate([[0, 1], [1, 0], [.8, .8]], 1)
    assert result['prediction'] == [.8, .8]
    assert result['accepted']
    assert result['prediction'] not in [[0, 1], [1, 0]]


def test_key_equivalence_and_target_do_not_change_attack():
    sample = {'id': 'x', 'replicas': [[0.], [0.], [0.]], 'key_ids': ['a', 'b', 'c']}
    a = ri.evaluate(sample, [0.], 'two-colluding', 1, .1, .01)
    b = ri.evaluate(sample, [10.], 'two-colluding', 1, .1, .01)
    assert a['prediction'] == b['prediction']
    assert a['max_absolute_error'] != b['max_absolute_error']
    assert ri.evaluate(sample, [0.], 'shared-key-disclosure', 1, .1, .01)['compromised_replicas'] == 1


def fixtures(tmp_path):
    obs = {'schema': 'llm_attacker.replica_observations.v1', 'samples': [
        {'id': 'x', 'replicas': [[0.], [0.], [0.]], 'key_ids': ['a', 'a', 'b']}]}
    truth = {'schema': 'llm_attacker.replica_truth.v1', 'targets': {'x': [0.]}}
    cp, tp = tmp_path/'captures.json', tmp_path/'truth.json'
    cp.write_text(json.dumps(obs))
    tp.write_text(json.dumps(truth))
    return cp, tp, obs, truth


def test_fixture_replay(tmp_path):
    cp, tp, _, _ = fixtures(tmp_path)
    out = tmp_path/'output.json'
    args = ri.build_parser().parse_args(['--captures', str(cp), '--truth', str(tp),
        '--arms', 'clean', 'two-colluding', 'shared-key-disclosure', '--output', str(out)])
    assert ri.run(args) == 0
    art = json.loads(out.read_text())
    assert art['evidence_tier'] == 'fixture-replay-engineering'
    assert art['results'][-1]['compromised_replicas'] == 2
    assert set(art['input_sha256']) == {'captures', 'truth'}


@pytest.mark.parametrize('mutation', ['nan', 'extra', 'duplicate', 'dimension', 'missing-truth', 'bool'])
def test_strict_fixture_failures(tmp_path, mutation):
    cp, tp, obs, truth = fixtures(tmp_path)
    if mutation == 'nan': obs['samples'][0]['replicas'][0][0] = float('nan')
    elif mutation == 'bool': obs['samples'][0]['replicas'][0][0] = True
    elif mutation == 'extra': obs['samples'][0]['unknown'] = 1
    elif mutation == 'duplicate': obs['samples'].append(obs['samples'][0])
    elif mutation == 'dimension': truth['targets']['x'] = [0, 1]
    else: truth['targets'] = {}
    cp.write_text(json.dumps(obs))
    tp.write_text(json.dumps(truth))
    with pytest.raises(ValueError): ri.load_inputs(cp, tp)


@pytest.mark.parametrize('flags', [[], ['--toy', '--tolerance', 'nan'], ['--toy', '--replicas', '2'], ['--toy', '--evidence']])
def test_invalid_configuration(flags):
    with pytest.raises(ValueError): ri.run(ri.build_parser().parse_args(flags))


def test_shift_zero_is_null_control_and_k5_tolerates_two():
    s = {'id': 'x', 'replicas': [[0.]]*5, 'key_ids': ['a']*5}
    assert not ri.evaluate(s, [0.], 'two-colluding', 1, .1, .01)['accepted_error']
    s['replicas'] = s['replicas'][:3]
    assert not ri.evaluate(s, [0.], 'two-colluding', 0, .1, .01)['accepted_error']


def test_duplicate_json_keys_rejected(tmp_path):
    cp, tp, _, _ = fixtures(tmp_path)
    tp.write_text('{"schema":"llm_attacker.replica_truth.v1","targets":{"x":[0],"x":[1]}}')
    with pytest.raises(ValueError, match='duplicate JSON key'):
        ri.load_inputs(cp, tp)


LIVE_ARMS = ['live-clean', 'live-one-malicious', 'live-two-colluding',
             'live-k2-disagreement', 'live-honest-heterogeneity',
             'live-shared-key-disclosure']


def _live_run(tmp_path, name, runner='inproc', seed=1234, extra=()):
    out = tmp_path / name
    args = ri.build_parser().parse_args(
        ['--toy', '--live', '--quick', '--channel-runner', runner,
         '--seed', str(seed), '--arms', *LIVE_ARMS, '--output', str(out),
         *extra])
    assert ri.run(args) == 0
    return json.loads(out.read_text())


def test_live_arms_match_fixture_verdicts(tmp_path):
    art = _live_run(tmp_path, 'live.json')
    assert art['evidence_tier'] == 'live-channel-engineering'
    s = {r['arm']: r for r in art['summary']}
    assert s['live-clean']['false_positive_rate'] == 0
    assert s['live-clean']['false_abort_rate'] == 0
    assert s['live-clean']['accepted_error_rate'] == 0
    assert s['live-one-malicious']['accepted_error_rate'] == 0
    assert s['live-two-colluding']['accepted_error_rate'] == 1
    assert s['live-two-colluding']['alarm_rate'] == 1
    assert s['live-k2-disagreement']['availability_abort_rate'] == 1
    assert s['live-honest-heterogeneity']['accepted_error_rate'] == 1
    rows = [r for r in art['results'] if r['arm'] == 'live-two-colluding']
    assert all(r['replica_count'] == 3 for r in rows)
    key_rows = [r for r in art['results']
                if r['arm'] == 'live-shared-key-disclosure']
    assert key_rows and all(r['compromised_replicas'] == 1 for r in key_rows)


def test_live_channels_are_independent(tmp_path):
    channels = ri.spawn_channels(3, 4, 99, 'inproc')
    outputs = [ch.respond(5, 1) for ch in channels]
    assert len({tuple(v) for v in outputs}) > 1
    assert len({ch.key_id for ch in channels}) == 3
    decision = ri.aggregate(outputs, .1)
    assert decision['accepted'] and not decision['alarm']
    target = ri._latent_target(5, 4)
    assert max(abs(a - b) for a, b in zip(decision['prediction'], target)) <= .01


def test_subprocess_worker_round_trip():
    ch = ri.spawn_channels(2, 3, 11, 'subprocess')
    try:
        first, second = ch[0].respond(5, 1), ch[0].respond(6, 1)
        inproc = ri.spawn_channels(2, 3, 11, 'inproc')
        inproc[0].respond(5, 1), inproc[0].respond(6, 1)
        assert ch[0].respond(7, 1) == inproc[0].respond(7, 1)
    finally:
        for c in ch:
            c.close()
    assert len(first) == len(second) == 3 and first != second
    assert ch[0].key_id == ri.channel_key_id(0, 11) != ch[1].key_id


def test_live_run_is_deterministic(tmp_path):
    a = _live_run(tmp_path, 'a.json', seed=77)
    b = _live_run(tmp_path, 'b.json', seed=77)
    assert a['summary'] == b['summary']
    assert a['results'] == b['results']
    c = _live_run(tmp_path, 'c.json', seed=78)
    assert c['results'] != a['results']


def test_live_arms_require_live_mode(tmp_path):
    with pytest.raises(ValueError, match='live'):
        ri.run(ri.build_parser().parse_args(['--toy', '--arms', 'live-clean']))
    cp, tp, _, _ = fixtures(tmp_path)
    with pytest.raises(ValueError, match='live'):
        ri.run(ri.build_parser().parse_args(
            ['--captures', str(cp), '--truth', str(tp), '--live',
             '--arms', 'clean']))
