#!/usr/bin/env python3
"""Run the frozen measurement-validation selection in an isolated CPU snapshot."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='A new, non-existent result directory; never overwritten.')
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selection_path = Path(__file__).with_name('selection.json')
    selection = json.loads(selection_path.read_text())
    commit = selection['source_commit']
    categories = selection['categories']
    selected = [node for category in categories for node in category['tests']]
    assert len(selected) == len(set(selected)) == 25
    roots = ['attacker', 'pyproject.toml', 'tests/conftest.py',
             'tests/conftest_plugin.py',
             'experiments/bisr_fidelity_predict.py',
             'experiments/analyze_validation.py',
             'experiments/validate_reconstruction.py',
             'experiments/validation_protocol.json']
    roots += sorted({node.split('::')[0] for node in selected})
    archive_command = ['git', '-C', str(repository), 'archive', commit, '--', *roots]
    archive = subprocess.run(archive_command, capture_output=True, check=True).stdout
    snapshot = Path(tempfile.mkdtemp(prefix='paper-b-measurement-20261001-'))
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as bundle:
        bundle.extractall(snapshot, filter='data')
    source_hashes = {str(path.relative_to(snapshot)): digest(path)
                     for path in sorted(snapshot.rglob('*')) if path.is_file()}
    (output / 'source-files.sha256').write_text(''.join(
        f'{sha}  {name}\n' for name, sha in source_hashes.items()))
    env_overrides = {
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': os.pathsep.join([str(snapshot / 'tests'), str(snapshot)]),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'PYTHONHASHSEED': '0',
    }
    env = {**os.environ, **env_overrides}
    command = [sys.executable, '-m', 'pytest', '-vv', '-ra',
               '-o', f'cache_dir={snapshot / ".pytest_cache"}',
               f'--junitxml={output / "junit.xml"}', *selected]
    versions = {name: importlib.metadata.version(name)
                for name in ['pytest', 'numpy', 'torch', 'transformers']}
    metadata = {
        'schema': 'llm-attacker.measurement-validation-run.v1',
        'start_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': commit, 'source_tag': selection['source_tag'],
        'selection_sha256': digest(selection_path),
        'reproduction_script_sha256': digest(Path(__file__)),
        'archive_sha256': hashlib.sha256(archive).hexdigest(),
        'archive_paths': roots, 'archive_command': archive_command,
        'snapshot': str(snapshot), 'python': sys.version,
        'executable': sys.executable, 'platform': platform.platform(),
        'machine': platform.machine(), 'cpu_count': os.cpu_count(),
        'versions': versions, 'environment_overrides': env_overrides,
        'command': command, 'selected_functions': len(selected),
    }
    write_json(output / 'run.json', metadata)
    packages = sorted(({'name': dist.metadata['Name'], 'version': dist.version}
                       for dist in importlib.metadata.distributions()),
                      key=lambda item: (item['name'].lower(), item['version']))
    write_json(output / 'packages.json', packages)
    (output / 'command.txt').write_text(
        'cd ' + shlex.quote(str(snapshot)) + '\n' +
        ' '.join(f'{key}={shlex.quote(value)}' for key, value in env_overrides.items()) +
        ' ' + shlex.join(command) + '\n')
    print(f'Source commit: {commit}', flush=True)
    print(f'Selection SHA-256: {metadata["selection_sha256"]}', flush=True)
    print(f'Snapshot: {snapshot}', flush=True)
    print(f'Dependencies: {json.dumps(versions, sort_keys=True)}', flush=True)
    print(f'Running {len(selected)} frozen test functions on CPU.', flush=True)
    with (output / 'pytest.stdout.txt').open('w') as stream:
        completed = subprocess.run(command, cwd=snapshot, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
    print((output / 'pytest.stdout.txt').read_text(), end='', flush=True)
    metadata.update(end_utc=datetime.now(timezone.utc).isoformat(),
                    pytest_returncode=completed.returncode,
                    source_unchanged=all(digest(snapshot / name) == sha
                                         for name, sha in source_hashes.items()
                                         if name not in ('tests/last_run.json', 'tests/last_run.md')))
    write_json(output / 'run.json', metadata)
    for suffix in ('json', 'md'):
        report = snapshot / 'tests' / f'last_run.{suffix}'
        if report.is_file():
            shutil.copyfile(report, output / f'pytest-record.{suffix}')
    record_path = output / 'pytest-record.json'
    if not record_path.is_file():
        raise RuntimeError('The frozen pytest recorder did not produce a result file.')
    record = json.loads(record_path.read_text())
    category_by_function = {node: category['id']
                            for category in categories for node in category['tests']}
    observed = []
    counts = {category['id']: Counter() for category in categories}
    for result in record['tests']:
        base = result['nodeid'].split('[')[0]
        category_id = category_by_function[base]
        observed.append({**result, 'category': category_id})
        counts[category_id][result['outcome']] += 1
    junit_cases = ET.parse(output / 'junit.xml').getroot().findall('.//testcase')
    summary = {
        'schema': 'llm-attacker.measurement-validation-results.v1',
        'source_commit': commit, 'selection_sha256': digest(selection_path),
        'returncode': completed.returncode, 'totals': record['totals'],
        'junit_cases': len(junit_cases),
        'category_counts': {key: {outcome: value.get(outcome, 0)
                                 for outcome in ('passed', 'failed', 'skipped')}
                            for key, value in counts.items()},
        'tests': observed,
    }
    write_json(output / 'results.json', summary)
    (output / 'collected-nodeids.txt').write_text(''.join(
        result['nodeid'] + '\n' for result in observed))
    if len(junit_cases) != record['totals']['tests']:
        raise RuntimeError('JUnit and recorder test counts disagree; inspect both reports.')
    if not metadata['source_unchanged']:
        raise RuntimeError('Archived source changed during execution.')
    print(json.dumps({'totals': summary['totals'],
                      'category_counts': summary['category_counts']}, indent=2), flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
