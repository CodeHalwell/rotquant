import copy
import json
import subprocess
import sys

import nbformat
import pytest

from scripts import build_qwen35_ternary_research_v2_notebook as notebook


def test_new_notebook_is_clean_deterministic_and_defaults_to_parity_with_immutable_history():
    actual = nbformat.read(notebook.OUTPUT, as_version=4)
    expected = notebook.build_notebook()
    assert actual == expected
    nbformat.validate(actual)
    sources = '\n'.join(cell.source for cell in actual.cells)
    assert 'STAGE = "parity"' in sources
    assert notebook.HISTORICAL_ROOT in sources
    assert 'rotquant/ternary_research_v2' in sources
    assert '--historical-root' in sources and '--stage' in sources
    assert 'run_qwen35_ternary_research_v2.py' in sources
    assert 'use_cache=True' not in sources
    assert 'Full comparison complete:' in sources and 'Cached-decoding parity complete:' in sources
    assert 'historical_path.read_text()' in sources and 'historical_path.write_text' not in sources
    for cell in actual.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None and cell.outputs == []
            compile(cell.source, cell.id, 'exec')


def test_reference_setup_and_exact_torch_identity_checks_remain_present():
    sources = '\n'.join(cell.source for cell in notebook.build_notebook().cells)
    for required in ('verify_source_bundle', 'unpack_source_bundle', 'torch-before.json',
                     'torch-after.json', 'torch-final.json', 'Fast Hadamard CUDA smoke passed',
                     'drive.mount("/content/drive")', 'files.upload()', 'files.download(str(archive))'):
        assert required in sources
    assert 'assert config_path.read_text() == text' in sources
    assert 'assert notebook_identity.read_text() == identity_text' in sources


def test_runtime_profile_phase_identities_do_not_overwrite_each_other_or_mutate_saved_rows():
    rows = [{'label': 'ternary@reference', 'model': 'ternary', 'profile': 'reference'},
            {'label': 'ternary@decode4', 'model': 'ternary', 'profile': 'decode4'}]
    original = copy.deepcopy(rows)
    fresh = notebook.comparison_group(rows, 'fresh')
    historical = notebook.comparison_group(rows[:1], 'historical')
    assert rows == original
    assert len({row['label'] for row in fresh + historical}) == 3
    assert '/ prism / historical' in notebook.row_identity({'label': 'bonsai2_4b'}, 'historical')
    with pytest.raises(ValueError, match='duplicate'):
        notebook.comparison_group(rows[:1] * 2, 'fresh')


def test_embedded_renderer_supports_partial_parity_without_inventing_measurements(tmp_path):
    report = {'stage': 'parity', 'complete': False, 'parity_complete': True,
              'historical_rows': [{'label': 'fp16', 'status': 'measured'}],
              'rows': [{'label': 'ternary@decode4', 'model': 'ternary', 'profile': 'decode4',
                        'phase': 'fresh', 'status': 'pending'}],
              'historical_import': {'sha256': 'original'}, 'gates': {'cache': {'passed': True}}}
    original = copy.deepcopy(report)
    script, source = tmp_path / 'render.py', tmp_path / 'comparison.json'
    script.write_text(notebook.presentation_script())
    source.write_text(json.dumps(report))
    subprocess.run([sys.executable, str(script), str(source), str(tmp_path)], check=True, timeout=30)
    result = json.loads((tmp_path / 'readout.json').read_text())
    assert report == original
    assert not result['complete'] and result['parity_complete']
    assert result['groups']['fresh']['tables']['models'][0]['Perplexity'] is None
    assert result['groups']['fresh']['figures'] == []
    assert result['groups']['historical']['tables']['models'][0]['Model'].endswith('/ historical')
    assert json.loads((tmp_path / 'readout.json').read_text())['historical_import'] == {'sha256': 'original'}
