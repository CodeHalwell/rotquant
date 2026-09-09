"""Pinned public-task inputs and inert answer checkers for the W5 release gate.

No benchmark program or model output is executed. Dataset text is data, never
an instruction to the host. This is a custom zero-shot/subset protocol, not an
official leaderboard submission or an assertion of pretraining cleanliness.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from decimal import Decimal
from pathlib import Path

DATASETS = {
    "gsm8k": {"repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866",
              "config": "main", "split": "test", "rows": 1319, "license": "MIT"},
    "cruxeval_o": {"repo": "cruxeval-org/cruxeval", "revision": "b96af0450242eb4da433032b90998f25588a5d0f",
                   "config": "default", "split": "test", "rows": 800, "license": "MIT"},
    "ifeval": {"repo": "google/IFEval", "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
               "config": "default", "split": "train", "rows": 541, "license": "Apache-2.0"},
}
BENCHMARKS = tuple(DATASETS)
SCORER_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
SCORER_FILES = {
    "instruction_following_eval/instructions.py": "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    "instruction_following_eval/instructions_util.py": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
    "instruction_following_eval/instructions_registry.py": "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    "instruction_following_eval/evaluation_lib.py": "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
NLTK_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_ZIP_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
SCORER_PACKAGES = {"nltk": "3.9.2", "langdetect": "1.0.9", "immutabledict": "4.2.1", "absl-py": "2.3.1"}
CAPS = {"gsm8k": 1024, "cruxeval_o": 512, "ifeval": 2048}
MAX_PROMPT_TOKENS = 2048
SELECTION_SEED = 20260909


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _fetch(url, digest):
    with urllib.request.urlopen(url, timeout=90) as response:
        value = response.read(16 * 1024 * 1024)
    if hashlib.sha256(value).hexdigest() != digest:
        raise ValueError(f"download hash mismatch: {url}")
    return value


def prepare_scorers(cache):
    """Fetch only four reviewed upstream modules + license and non-pickle data."""
    cache = Path(cache)
    base = f"https://raw.githubusercontent.com/google-research/google-research/{SCORER_REVISION}/"
    for name, sha in SCORER_FILES.items():
        path = cache / name
        if path.exists():
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f"scorer cache changed: {path}")
            continue
        print(f"Download pinned IFEval checker: {name}", flush=True)
        value = _fetch(base + name, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    url = f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_REVISION}/packages/tokenizers/punkt_tab.zip"
    archive = cache / "punkt_tab.zip"
    if not archive.exists():
        print("Download pinned NLTK sentence tables", flush=True)
        archive.write_bytes(_fetch(url, NLTK_ZIP_SHA256))
    if archive.is_symlink() or hashlib.sha256(archive.read_bytes()).hexdigest() != NLTK_ZIP_SHA256:
        raise ValueError("NLTK archive changed")
    # Only the English plain-text tables used by Google's checker. No pickle,
    # dynamic NLTK download, arbitrary archive path, or executable input.
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as bundle:
        for name in bundle.namelist():
            if not name.startswith("punkt_tab/english/") or name.endswith("/"):
                continue
            relative = Path(name)
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError("unsafe archive member")
            path = cache / "nltk_data/tokenizers" / relative
            value = bundle.read(name)
            if path.exists() and (path.is_symlink() or path.read_bytes() != value):
                raise ValueError("NLTK table changed")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(value)
    return scorer_identity(cache)


def scorer_identity(cache):
    cache = Path(cache).resolve()
    for package, version in SCORER_PACKAGES.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"install {package}=={version} for this protocol")
    files = {}
    for name, expected in SCORER_FILES.items():
        path = cache / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_symlink() or actual != expected:
            raise ValueError(f"scorer hash mismatch: {name}")
        files[name] = actual
    archive = cache / "punkt_tab.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != NLTK_ZIP_SHA256:
        raise ValueError("NLTK archive changed")
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name.startswith("punkt_tab/english/") and not name.endswith("/"):
                path = cache / "nltk_data/tokenizers" / name
                value = bundle.read(name)
                if path.is_symlink() or path.read_bytes() != value:
                    raise ValueError("NLTK table changed")
                files[name] = hashlib.sha256(value).hexdigest()
    return {"google_revision": SCORER_REVISION, "packages": SCORER_PACKAGES,
            "files": files, "nltk_revision": NLTK_REVISION,
            "langdetect_seed": SELECTION_SEED, "random_seed": "fixed independently per example and strict/loose pass"}


class IFEvalScorer:
    def __init__(self, cache):
        self.identity = scorer_identity(cache)
        cache = Path(cache).resolve()
        import nltk

        # Force the hashed tables, never a machine's unrelated punkt installation.
        nltk.data.path[:] = [str(cache / "nltk_data")]
        sys.path.insert(0, str(cache))
        self.lib = importlib.import_module("instruction_following_eval.evaluation_lib")
        for name, module in list(sys.modules.items()):
            if (name.startswith("instruction_following_eval.") and getattr(module, "__file__", None)
                    and not Path(module.__file__).resolve().is_relative_to(cache)):
                raise ValueError("an unrelated IFEval module is already imported")
        from instruction_following_eval import instructions_util

        if instructions_util.count_sentences("First sentence. Second sentence.") != 2:
            raise ValueError("sentence-checker preflight failed")
        known = {"key": -1, "prompt": "Include cobalt.",
                 "instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["cobalt"]}]}
        if not self.score(known, "cobalt")["prompt_strict"] or self.score(known, "amber")["prompt_strict"]:
            raise ValueError("instruction-checker positive/negative preflight failed")

    def score(self, row, text):
        from langdetect import DetectorFactory

        example = self.lib.InputExample(
            key=row["key"], instruction_id_list=row["instruction_id_list"], prompt=row["prompt"],
            kwargs=[{k: v for k, v in entry.items() if v is not None} for entry in row["kwargs"]],
        )
        if not example.instruction_id_list or len(example.instruction_id_list) != len(example.kwargs):
            raise ValueError("missing IFEval oracle")
        state, detector_seed = random.getstate(), DetectorFactory.seed
        try:
            outputs = {}
            for mode in ("strict", "loose"):
                random.seed(f"{SELECTION_SEED}:{row['key']}")
                DetectorFactory.seed = SELECTION_SEED
                check = getattr(self.lib, f"test_instruction_following_{mode}")
                out = check(example, {example.prompt: text})
                outputs[f"prompt_{mode}"] = bool(out.follow_all_instructions)
                outputs[f"instructions_{mode}"] = [bool(v) for v in out.follow_instruction_list]
            return outputs
        finally:
            random.setstate(state)
            DetectorFactory.seed = detector_seed


def literal(text):
    """Bounded Python literal parser. Never eval/exec or execute dataset code."""
    if len(text) > 65536:
        raise ValueError("literal too long")
    tree = ast.parse(text.strip(), mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 8192 or any(isinstance(n, (ast.Call, ast.BinOp)) for n in nodes):
        raise ValueError("not a bounded literal")
    for node in nodes:
        if isinstance(node, ast.Dict):
            keys = [ast.literal_eval(k) for k in node.keys]
            if len(set(keys)) != len(keys):
                raise ValueError("duplicate dictionary key")
    return ast.literal_eval(tree)


def same_literal(left, right):
    def typed(value):
        if isinstance(value, dict):
            return dict, frozenset((typed(k), typed(v)) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return type(value), tuple(typed(v) for v in value)
        if isinstance(value, set):
            return set, frozenset(typed(v) for v in value)
        return type(value), value

    return typed(left) == typed(right)


def numeric_answer(text):
    # Require a terminal answer line. Reasoning digits cannot accidentally pass.
    match = re.search(r"(?:^|\n)####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\Z", text.strip())
    return None if match is None else Decimal(match.group(1).replace(",", ""))


def score(item, text, truncated=False, ifeval=None):
    if item["benchmark"] == "gsm8k":
        answer = numeric_answer(text)
        valid = answer is not None
        passed = valid and answer == Decimal(item["answer"])
        details = {"answer_format_valid": valid}
    elif item["benchmark"] == "cruxeval_o":
        matches = re.findall(r"\[ANSWER\](.*?)\[/ANSWER\]", text, flags=re.DOTALL)
        valid, passed = False, False
        if len(matches) == 1:
            try:
                parsed = literal(matches[0])
                passed = same_literal(parsed, literal(item["answer"]))
                valid = True
            except (ValueError, TypeError, SyntaxError, RecursionError):
                pass
        details = {"answer_format_valid": valid}
    elif item["benchmark"] == "ifeval":
        if ifeval is None:
            raise ValueError("IFEval scorer is required; never silently skip instructions")
        details = ifeval.score(item["oracle"], text)
        passed = details["prompt_strict"]
    else:
        raise ValueError("unknown benchmark")
    return {"task_success": bool(passed and not truncated), "truncated": bool(truncated),
            "checker_success_before_truncation_guard": bool(passed), **details}


def make_item(name, row, index):
    if name == "gsm8k":
        answer = numeric_answer(row["answer"])
        if answer is None:
            raise ValueError("unsupported GSM8K gold answer")
        prompt = row["question"] + "\n\nSolve the problem. End with a line containing #### followed by only the numeric answer."
        payload = {"answer": str(answer)}
        key = str(index)
    elif name == "cruxeval_o":
        literal(row["output"])
        prompt = ("Predict the exact return value of the Python function for the supplied arguments. "
                  "Return a Python literal between [ANSWER] and [/ANSWER]. Do not write or run a program.\n\n"
                  + row["code"] + "\n\nf(" + row["input"] + ")")
        payload, key = {"answer": row["output"]}, row["id"]
    elif name == "ifeval":
        prompt, key = row["prompt"], str(row["key"])
        payload = {"oracle": row}
    else:
        raise ValueError("unknown benchmark")
    return {"id": name + "-" + key, "benchmark": name, "kind": "task", "source_row": index,
            "source_row_sha256": fingerprint(row), "prompt": prompt, "max_new_tokens": CAPS[name], **payload}


def load_items(count):
    """count=0 selects the entire registered split, never model-dependent filtering."""
    from datasets import load_dataset

    if type(count) is not int or count < 0 or count > min(d["rows"] for d in DATASETS.values()):
        raise ValueError("samples must be 0 (all) or between 1 and 541 per benchmark")
    output, receipts = [], {}
    for name, spec in DATASETS.items():
        print(f"Load pinned {name}: {spec['revision']}", flush=True)
        data = load_dataset(spec["repo"], spec["config"], split=spec["split"], revision=spec["revision"])
        if len(data) != spec["rows"]:
            raise ValueError(f"unexpected source row count: {name}")
        all_items = [make_item(name, dict(row), i) for i, row in enumerate(data)]
        if len({v["id"] for v in all_items}) != len(all_items):
            raise ValueError("duplicate source IDs")
        order = sorted(all_items, key=lambda v: fingerprint([SELECTION_SEED, v["id"]]))
        chosen = order if count == 0 else order[:count]
        receipts[name] = {**spec, "selected": len(chosen),
                          "ordered_source_rows_sha256": fingerprint([v["source_row_sha256"] for v in all_items])}
        output.extend(chosen)
    return output, receipts
