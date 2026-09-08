"""Fresh authored diagnostics, strict inert oracles and clustered comparisons.

These are synthetic unit tasks, not a public coding/agentic benchmark. Variants
share a family so uncertainty never treats a template expansion as independent.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def task_suite():
    tasks = []

    def add(domain, family, variant, prompt, expected):
        tasks.append(
            {
                "id": f"{domain}-{family}-{variant}",
                "domain": domain,
                "family": f"{domain}-{family}",
                "kind": "task",
                "prompt": prompt,
                "expected": expected,
            }
        )

    for i in range(6):
        a, b = 13 + i * 7, 3 + i
        cases = [
            (
                "filter",
                f"xs = {list(range(a, a + 5))}\nprint(sum(x for x in xs if x % 2 == 0))",
                sum(x for x in range(a, a + 5) if x % 2 == 0),
            ),
            (
                "slice",
                f"xs = {list(range(a, a + 6))}\nprint(xs[-1:0:-2])",
                list(range(a, a + 6))[-1:0:-2],
            ),
            ("dict", f"d = {{'a': {a}, 'b': {b}}}\nd['a'] += d.pop('b')\nprint(d['a'])", a + b),
            ("loop", f"x = {a}\nfor j in range({b}):\n    x += 2*j\nprint(x)", a + b * (b - 1)),
        ]
        for family, snippet, expected in cases:
            add(
                "code_trace",
                family,
                i,
                "What does this Python 3 program print? Return only "
                'JSON of the form {"answer": VALUE}; encode lists as JSON arrays.\n' + snippet,
                {"answer": expected},
            )
        for family, prompt, expected in [
            (
                "sort",
                (
                    f"Sort these IDs numerically ascending: {[a + 2, a, a + 1]}. "
                    'Return only {"ids": [integers]}.'
                ),
                {"ids": [a, a + 1, a + 2]},
            ),
            (
                "extract",
                (
                    f"Record: name=Beacon-{i}; quantity={a}; active=false. "
                    "Return only JSON with name (string), quantity (integer), active (boolean)."
                ),
                {"name": f"Beacon-{i}", "quantity": a, "active": False},
            ),
            (
                "null",
                (
                    f"Record: item=R-{a}; location is missing. "
                    "Return only JSON with item and location; use null for missing values."
                ),
                {"item": f"R-{a}", "location": None},
            ),
            (
                "aggregate",
                (
                    f"Orders have quantities {[a, b, a]}. Return only JSON with "
                    "count (number of orders) and total (sum of quantities)."
                ),
                {"count": 3, "total": 2 * a + b},
            ),
        ]:
            add("structured", family, i, prompt, expected)
        for family, prompt, expected in [
            (
                "read_only",
                (
                    f"Tools: read_record(id), delete_record(id). User: inspect record R-{a}, "
                    "do not delete it."
                ),
                {"name": "read_record", "arguments": {"id": f"R-{a}"}},
            ),
            (
                "condition",
                (
                    f"Tools: alert(sensor). Alert only if temperature > 50. Sensor S-{i} "
                    f"has temperature {a}."
                ),
                {"name": "alert", "arguments": {"sensor": f"S-{i}"}}
                if a > 50
                else {"name": "none", "arguments": {}},
            ),
            (
                "missing",
                (
                    f"Tools: book_trip(destination, date). User: book a trip to Porto-{i}, "
                    "but no date was provided. Missing arguments must not be invented; choose none."
                ),
                {"name": "none", "arguments": {}},
            ),
            (
                "lookup",
                (
                    f"Tools: search(query), send_email(to, body). User: search for invoice INV-{a}. "
                    "Do not send email."
                ),
                {"name": "search", "arguments": {"query": f"INV-{a}"}},
            ),
        ]:
            add(
                "tool_selection",
                family,
                i,
                prompt + " Return only one JSON object with exactly "
                'name and arguments. If no tool should be called use {"name":"none","arguments":{}}.',
                expected,
            )
    languages = {
        "es": 'Calcula {a} + {b}. Responde únicamente con JSON: {{"answer": número}}.',
        "fr": 'Calcule {a} + {b}. Réponds uniquement en JSON : {{"answer": nombre}}.',
        "de": 'Berechne {a} + {b}. Antworte nur mit JSON: {{"answer": Zahl}}.',
        "ja": '{a} + {b} を計算してください。JSON {{"answer": 数値}} だけを返してください。',
        "ar": 'احسب {a} + {b}. أجب فقط بصيغة JSON: {{"answer": عدد}}.',
        "hi": '{a} + {b} की गणना करें। केवल JSON दें: {{"answer": संख्या}}।',
    }
    for language, template in languages.items():
        for i in range(4):
            a, b = 37 + i * 13, 19 + i * 11
            add(
                "multilingual",
                "addition",
                f"{language}-{i}",
                template.format(a=a, b=b),
                {"answer": a + b},
            )
            tasks[-1]["language"] = language
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks):
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique and nonempty")
    if len({t["prompt"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate task prompt")
    for task in tasks:
        if not all(task.get(key) for key in ("id", "domain", "family", "prompt")):
            raise ValueError("task metadata missing")
        if not isinstance(task.get("expected"), dict):
            raise TypeError("task must have a JSON-object oracle")


def score_task(text, expected, *, truncated=False):
    """Strict JSON, exact types/keys; no eval, execution, tools or repair."""

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def invalid(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def same(actual, desired):
        if type(actual) is not type(desired):
            return False
        if isinstance(actual, dict):
            return actual.keys() == desired.keys() and all(
                same(actual[k], v) for k, v in desired.items()
            )
        if isinstance(actual, list):
            return len(actual) == len(desired) and all(same(a, b) for a, b in zip(actual, desired))
        return actual == desired

    try:
        actual = json.loads(text.strip(), object_pairs_hook=unique, parse_constant=invalid)
        valid = isinstance(actual, dict)
        return {
            "json_valid": valid,
            "task_success": valid and same(actual, expected) and not truncated,
            "truncated": bool(truncated),
        }
    except (ValueError, TypeError):
        return {"json_valid": False, "task_success": False, "truncated": bool(truncated)}


def paired_family_interval(left, right, field, *, draws=4000, seed=20260908):
    """Right minus left; equal-family bootstrap, not token IID or seed replication."""
    if not left or {r["id"] for r in left} != {r["id"] for r in right}:
        raise ValueError("paired records must contain identical IDs")
    if len({r["id"] for r in left}) != len(left) or len(left) != len(right):
        raise ValueError("duplicate paired IDs")
    other = {r["id"]: r for r in right}
    grouped = {}
    for row in left:
        peer = other[row["id"]]
        if (row["input_hash"], row["family"], row["tokens"]) != (
            peer["input_hash"],
            peer["family"],
            peer["tokens"],
        ):
            raise ValueError("paired input/family/denominator mismatch")
        delta = float(peer[field]) - float(row[field])
        if not np.isfinite(delta):
            raise ValueError("non-finite paired metric")
        grouped.setdefault(row["family"], []).append(delta)
    values = np.array([np.mean(grouped[key]) for key in sorted(grouped)])
    result = {
        "delta": float(values.mean()),
        "families": len(values),
        "prompts": len(left),
        "weighting": "equal family means",
        "conditional_on": "one frozen diagnostic set; not a seed CI",
    }
    if len(values) < 2:
        return {**result, "ci95": None, "reason": "fewer than two independent authored families"}
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return {**result, "ci95": np.quantile(samples, [0.025, 0.975]).tolist(), "draws": draws}
