"""Engine-neutral competitive result ingestion and paired reporting."""

import copy

import pytest

from rotquant.eval.competition import REGISTERED_DOMAIN_QUOTAS
from rotquant.eval.competitive_run import (
    ArtifactFile,
    PromptObservation,
    RunFailure,
    RunMetadata,
    aggregate_competitive_run,
    artifact_identity,
    compare_run_reports,
    inspect_artifact_files,
)
from rotquant.eval.data_manifest import (
    DatasetManifest,
    DatasetSource,
    ManifestItem,
    protocol_from_manifests,
)


def _source():
    return DatasetSource(
        source_id="fixture",
        revision="v1",
        split="test",
        licenses=("MIT",),
        url="https://example.com/fixture",
    )


def _item(item_id, domain, token):
    return ManifestItem(
        item_id=item_id,
        domain=domain,
        source_id="fixture",
        source_record_id=item_id,
        token_ids=(token,),
        licenses=("MIT",),
    )


def _manifests_and_protocol():
    source = _source()
    calibration = DatasetManifest(
        role="calibration",
        tokenizer_id="org/tokenizer",
        tokenizer_revision="tokenizer-v1",
        chat_template_sha256="c" * 64,
        sources=(source,),
        items=(_item("calibration", "general", 1),),
        transformations=("fixture",),
        seed=17,
    )
    items = []
    token = 1_000
    for domain, count in REGISTERED_DOMAIN_QUOTAS.items():
        for index in range(count):
            items.append(_item(f"{domain}-{index}", domain, token))
            token += 1
    evaluation = DatasetManifest(
        role="evaluation",
        tokenizer_id="org/tokenizer",
        tokenizer_revision="tokenizer-v1",
        chat_template_sha256="c" * 64,
        sources=(source,),
        items=tuple(items),
        transformations=("fixture",),
        seed=17,
    )
    protocol = protocol_from_manifests(
        model_id="Qwen/Qwen3.5-4B",
        model_revision="model-v1",
        calibration=calibration,
        evaluation=evaluation,
    )
    return evaluation, protocol


def _metadata(protocol, *, name="rotquant", size=1_000_000):
    artifact_file = ArtifactFile(
        name=f"{name}.gguf",
        sha256=("a" if name == "rotquant" else "b") * 64,
        bytes=size,
    )
    return RunMetadata(
        name=name,
        format="gguf",
        artifact_sha256=artifact_file.sha256,
        artifact_bytes=size,
        artifact_files=(artifact_file,),
        engine="llama.cpp",
        engine_revision="engine-v1",
        protocol_fingerprint=protocol.fingerprint,
    )


def _observations(
    manifest,
    *,
    kl=0.1,
    top1=True,
    mismatch_at=None,
):
    source = tuple(range(32))
    observations = []
    for item in manifest.items:
        candidate = list(source)
        if mismatch_at is not None:
            candidate[mismatch_at] = 999
        observations.append(
            PromptObservation(
                item_sha256=item.token_sha256,
                domain=item.domain,
                teacher_kl=(kl,) * 32,
                top1_matches=(top1,) * 32,
                source_continuation=source,
                candidate_continuation=tuple(candidate),
            )
        )
    return tuple(observations)


def _completed_report(
    manifest,
    protocol,
    *,
    name="rotquant",
    size=1_000_000,
    kl=0.1,
    top1=True,
    mismatch_at=None,
):
    return aggregate_competitive_run(
        protocol=protocol,
        prompt_manifest=manifest,
        metadata=_metadata(protocol, name=name, size=size),
        observations=_observations(
            manifest,
            kl=kl,
            top1=top1,
            mismatch_at=mismatch_at,
        ),
    )


def test_completed_run_has_overall_domain_and_prompt_metrics():
    manifest, protocol = _manifests_and_protocol()
    report = _completed_report(manifest, protocol)
    assert RunMetadata.from_manifest(_metadata(protocol).manifest()) == _metadata(protocol)

    assert report["status"] == "completed"
    assert report["observed_prompt_count"] == 300
    assert report["failure_counts"] == {}
    assert len(report["per_prompt"]) == 300
    assert len(report["per_prompt"][0]["source_continuation_sha256"]) == 64
    assert set(report["domain_summaries"]) == set(REGISTERED_DOMAIN_QUOTAS)
    assert report["domain_summaries"]["agentic"]["prompt_count"] == 60
    evaluation = report["artifact_evaluation"]
    assert evaluation["scored_tokens"] == 9_600
    assert evaluation["mean_teacher_kl"] == pytest.approx(0.1)
    assert evaluation["max_teacher_kl"] == pytest.approx(0.1)
    assert evaluation["top1_agreement"] == 1.0
    assert evaluation["exact_trajectory_rate"] == 1.0
    assert evaluation["mean_matching_prefix"] == 32.0


def test_failure_is_distinct_from_quality_and_cannot_be_certified():
    manifest, protocol = _manifests_and_protocol()
    observations = _observations(manifest)[:-1]
    failure = RunFailure(
        item_sha256=manifest.items[-1].token_sha256,
        stage="generation",
        error_type="OutOfMemoryError",
        message="device exhausted",
    )
    report = aggregate_competitive_run(
        protocol=protocol,
        prompt_manifest=manifest,
        metadata=_metadata(protocol),
        observations=observations,
        failures=(failure,),
    )

    assert report["status"] == "incomplete"
    assert report["failure_counts"] == {"generation": 1}
    assert report["missing_item_sha256"] == []
    assert "artifact_evaluation" not in report
    assert "domain_summaries" not in report
    with pytest.raises(ValueError, match="incomplete"):
        compare_run_reports(
            candidate_report=report,
            baseline_report=report,
            protocol=protocol,
        )


def test_unreported_missing_prompt_is_explicit():
    manifest, protocol = _manifests_and_protocol()
    report = aggregate_competitive_run(
        protocol=protocol,
        prompt_manifest=manifest,
        metadata=_metadata(protocol),
        observations=_observations(manifest)[:-1],
    )
    assert report["status"] == "incomplete"
    assert report["missing_item_sha256"] == [manifest.items[-1].token_sha256]


def test_observation_identity_domain_and_length_are_validated():
    manifest, protocol = _manifests_and_protocol()
    observations = list(_observations(manifest))
    observations[0] = PromptObservation(
        item_sha256=observations[0].item_sha256,
        domain="wrong",
        teacher_kl=observations[0].teacher_kl,
        top1_matches=observations[0].top1_matches,
        source_continuation=observations[0].source_continuation,
        candidate_continuation=observations[0].candidate_continuation,
    )
    with pytest.raises(ValueError, match="domain does not match"):
        aggregate_competitive_run(
            protocol=protocol,
            prompt_manifest=manifest,
            metadata=_metadata(protocol),
            observations=tuple(observations),
        )

    short = PromptObservation(
        item_sha256=manifest.items[0].token_sha256,
        domain=manifest.items[0].domain,
        teacher_kl=(0.1,),
        top1_matches=(True,),
        source_continuation=(1,),
        candidate_continuation=(1,),
    )
    with pytest.raises(ValueError, match="per generated token"):
        aggregate_competitive_run(
            protocol=protocol,
            prompt_manifest=manifest,
            metadata=_metadata(protocol),
            observations=(short,),
        )


def test_paired_comparison_is_size_matched_domain_aware_and_deterministic():
    manifest, protocol = _manifests_and_protocol()
    candidate = _completed_report(
        manifest,
        protocol,
        kl=0.1,
        top1=True,
        mismatch_at=None,
        size=995_000,
    )
    baseline = _completed_report(
        manifest,
        protocol,
        name="baseline",
        kl=0.2,
        top1=False,
        mismatch_at=0,
    )
    first = compare_run_reports(
        candidate_report=candidate,
        baseline_report=baseline,
        protocol=protocol,
        bootstrap_draws=100,
        bootstrap_seed=9,
    )
    second = compare_run_reports(
        candidate_report=candidate,
        baseline_report=baseline,
        protocol=protocol,
        bootstrap_draws=100,
        bootstrap_seed=9,
    )

    assert first == second
    assert first["interpretation"].startswith("candidate minus baseline")
    assert first["artifact_comparison"]["size_delta_fraction"] == pytest.approx(-0.005)
    assert first["paired_prompt_deltas"]["mean_teacher_kl"]["mean_delta"] == pytest.approx(-0.1)
    assert first["paired_prompt_deltas"]["top1_agreement"]["mean_delta"] == 1.0
    assert first["domain_deltas"]["math"]["exact_trajectory"] == 1.0
    assert first["worst_domain_deltas"]["mean_teacher_kl"]["mean_delta"] == pytest.approx(-0.1)
    assert first["candidate_metadata"]["engine"] == "llama.cpp"
    assert len(first["per_prompt_deltas"]) == 300


def test_paired_comparison_rejects_size_mismatch():
    manifest, protocol = _manifests_and_protocol()
    candidate = _completed_report(manifest, protocol, size=2_000_000)
    baseline = _completed_report(manifest, protocol, name="baseline")
    with pytest.raises(ValueError, match="not size matched"):
        compare_run_reports(
            candidate_report=candidate,
            baseline_report=baseline,
            protocol=protocol,
            bootstrap_draws=10,
        )


def test_paired_comparison_rejects_post_aggregation_edits():
    manifest, protocol = _manifests_and_protocol()
    candidate = _completed_report(manifest, protocol)
    baseline = _completed_report(manifest, protocol, name="baseline")
    tampered = copy.deepcopy(candidate)
    tampered["per_prompt"][0]["mean_teacher_kl"] = 99.0
    with pytest.raises(ValueError, match="does not match the report contents"):
        compare_run_reports(
            candidate_report=tampered,
            baseline_report=baseline,
            protocol=protocol,
            bootstrap_draws=10,
        )


def test_artifact_files_are_measured_and_bundle_identity_is_order_independent(tmp_path):
    first_path = tmp_path / "weights.gguf"
    second_path = tmp_path / "projector.gguf"
    first_path.write_bytes(b"weights")
    second_path.write_bytes(b"projector")
    files = inspect_artifact_files(
        (("weights.gguf", first_path), ("projector.gguf", second_path))
    )

    assert sum(file.bytes for file in files) == len(b"weightsprojector")
    assert artifact_identity(files) == artifact_identity(tuple(reversed(files)))
    assert artifact_identity((files[0],)) == files[0].sha256


def test_run_metadata_rejects_unaccounted_artifact_bytes():
    _manifest, protocol = _manifests_and_protocol()
    artifact_file = ArtifactFile(name="weights.gguf", sha256="a" * 64, bytes=99)
    with pytest.raises(ValueError, match="sum of artifact files"):
        RunMetadata(
            name="rotquant",
            format="gguf",
            artifact_sha256=artifact_file.sha256,
            artifact_bytes=100,
            artifact_files=(artifact_file,),
            engine="llama.cpp",
            engine_revision="engine-v1",
            protocol_fingerprint=protocol.fingerprint,
        )
