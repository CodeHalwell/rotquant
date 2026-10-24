import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from scripts.run_qwen35_ternary_research_v2 import load_config
from scripts.ternary_research_v2_plan import DEFAULT_CONFIG
from scripts.ternary_research_v2_worker import V2Worker


@pytest.mark.parametrize('serialize', [yaml.safe_dump, json.dumps], ids=['yaml', 'notebook-json'])
def test_v2_config_preserves_learning_rates_through_cli_and_worker(tmp_path, monkeypatch, serialize):
    monkeypatch.delenv('ROTQUANT_TOKEN_CACHE_DIR', raising=False)
    expected = yaml.safe_load(DEFAULT_CONFIG.read_text())
    expected['v2']['global']['lr'] = 1e-5
    path = tmp_path / 'config.yaml'
    path.write_text(serialize(expected))
    config = load_config(path)
    assert config == expected
    assert type(config['v2']['global']['lr']) is float
    worker_path = tmp_path / 'config.json'
    worker_path.write_text(json.dumps(config))
    worker = V2Worker(SimpleNamespace(config=worker_path, output=tmp_path / 'stage',
                                     root=tmp_path, data=None))
    assert worker.config == expected
    result = subprocess.run([sys.executable, '-B', '-m', 'scripts.run_qwen35_ternary_research_v2',
        '--config', str(path), '--historical-root', str(tmp_path / 'history'),
        '--output-dir', str(tmp_path / 'results'), '--stage', 'parity', '--dry-run'],
        check=True, text=True, capture_output=True, timeout=30)
    assert json.loads(result.stdout)['stage'] == 'parity'
    assert not (tmp_path / 'results').exists()


def test_v2_config_does_not_coerce_explicit_text_learning_rates(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config['v2']['global']['lr'] = '1e-5'
    path = tmp_path / 'config.yaml'
    path.write_text(json.dumps(config))
    with pytest.raises(TypeError):
        load_config(path)
