"""Result records must survive interrupted notebook writes."""
import json

from rotquant.utils import write_result


def test_write_result_atomically_replaces_destination(tmp_path, monkeypatch):
    destination = tmp_path / "result.json"
    destination.write_text('{"old": true}')
    observed = {}

    from rotquant import utils

    real_replace = utils.os.replace

    def checked_replace(source, target):
        observed["temporary_exists"] = utils.os.path.exists(source)
        observed["target"] = target
        real_replace(source, target)

    monkeypatch.setattr(utils.os, "replace", checked_replace)
    write_result(str(destination), {"new": [1, 2, 3]})

    assert observed == {
        "temporary_exists": True,
        "target": str(destination),
    }
    assert json.loads(destination.read_text()) == {"new": [1, 2, 3]}
    assert not list(tmp_path.glob("*.tmp"))
