// Offline followup3 evidence audit; no model execution or CUDA replay.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import cp from 'node:child_process';
import assert from 'node:assert/strict';

const root = path.resolve(process.argv[2]);
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const filehash = p => hash(fs.readFileSync(path.join(root, p)));
const stagepath = (name, file = 'report.json') => `stages/${name}/attempt-001/${file}`;
const stage = (name, file) => read(stagepath(name, file));
const sum = xs => xs.reduce((a, b) => a + b, 0);
const near = (a, b) => assert.ok(Number.isFinite(a) && Number.isFinite(b) &&
  Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(b)), `${a} != ${b}`);
const workflow = read('workflow.json'), summary = read('summary.json');
const comparison = read('comparison-b5_v6_s0.json');
const revision = workflow.controls.revision;
assert.equal(revision, '445a436e9f0c4ddc4f8ba9c2e7955e38dd837b06');
assert.equal(workflow.controls.dirty_local_test, false);
assert.equal(workflow.status, 'passed');
assert.equal(summary.status, 'passed');
assert.equal(summary.error, null);
assert.equal(summary.stages.length, 20);
assert.equal(workflow.attempts.length, 20);
assert.equal(new Set(summary.stages.map(x => x.name)).size, 20);
for (const [i, row] of summary.stages.entries()) {
  assert.equal(row.status, 'passed');
  assert.equal(row.name, workflow.attempts[i].name);
  assert.equal(workflow.attempts[i].status, 'passed');
  near(row.elapsed_seconds, workflow.attempts[i].elapsed_seconds);
}
near(sum(summary.stages.map(x => x.elapsed_seconds)), workflow.active_seconds);
near(workflow.active_seconds / 60, summary.active_minutes);
assert.deepEqual(workflow.controls.conventional_baselines, []);
assert.equal(workflow.controls.diagnostic_profiles, false);
assert.equal(comparison.completed, true);
assert.equal(comparison.candidate, 'decode4');
assert.equal(comparison.comparisons.length, 2);
assert.deepEqual(comparison.profiles, []);
assert.equal(comparison.environment.device, 'NVIDIA A100-SXM4-40GB');

const remoteRoot = workflow.attempts[0].directory.split('/stages/')[0];
let included = 0;
const absent = new Set();
for (const attempt of workflow.attempts) {
  for (const [name, expected] of Object.entries(attempt.artifacts || {})) {
    if (!name.startsWith(remoteRoot + '/')) { absent.add(name); continue; }
    const relative = name.slice(remoteRoot.length + 1);
    assert.ok(path.resolve(root, relative).startsWith(root + path.sep));
    assert.equal(filehash(relative), expected.sha256, relative);
    assert.equal(fs.statSync(path.join(root, relative)).size, expected.bytes, relative);
    included++;
  }
}
const runtime = stage('binding-load', 'load.json').runtime_files;
const build = stage('native-build-and-load', 'build-receipt.json');
assert.equal(stage('native-build-and-load', 'cache.json').cache_hit, true);
assert.equal(runtime['librotquant_ggml_test.so'], build.library_sha256);
const code = {...workflow.controls.workflow_sources, ...build.external_sources};
for (const [name, sha] of Object.entries(code)) {
  assert.equal(hash(cp.execFileSync('git', ['show', `${revision}:${name}`])), sha, name);
}
assert.equal(filehashFromGit('integrations/llama.cpp/rotquant-native-v2.patch'), build.patch_sha256);
function filehashFromGit(name) {
  return hash(cp.execFileSync('git', ['show', `${revision}:${name}`]));
}
// Native sources also match the older build producer, explaining safe cache reuse.
for (const [name, sha] of Object.entries(build.external_sources)) {
  assert.equal(hash(cp.execFileSync('git', ['show', `${build.root_revision}:${name}`])), sha, name);
}

function checkBinding(binding) {
  assert.equal(binding.lookup, 'execution-library dependency handle');
  assert.equal(path.basename(binding.provider), 'libggml-cuda.so.0');
  assert.equal(binding.provider_sha256, runtime['libggml-cuda.so.0']);
  assert.equal(path.dirname(binding.provider), path.dirname(binding.library));
}
function checkDispatch(r) {
  assert.equal(r.passed, true);
  assert.equal(r.exact_reference, true);
  assert.deepEqual(r.runtime_files, runtime);
  checkBinding(r.binding);
  assert.deepEqual(r.cases.map(x => [x.kernel, x.tokens]),
    [['reference', 1], ['decode4', 1], ['reference', 4], ['decode4', 4]]);
  for (const c of r.cases) {
    assert.equal(c.passed, true);
    assert.equal(c.delta.instrumented, false);
    assert.equal(c.delta.operators.matrix.host_dispatches, 1);
    assert.equal(c.delta.decode_host_dispatches, c.kernel === 'decode4' && c.tokens === 1 ? 1 : 0);
    assert.equal(c.delta.tiled_host_dispatches, c.kernel === 'decode4' && c.tokens === 4 ? 1 : 0);
  }
}
checkDispatch(stage('candidate-dispatch-preflight'));
const candidateOps = stage('decode4-operators-CUDA0');
checkDispatch(candidateOps.dispatch_probe);
for (const name of ['operators-CPU', 'operators-CUDA0', 'decode4-operators-CUDA0']) {
  const r = stage(name);
  assert.equal(r.passed, true);
  assert.deepEqual(r.runtime_files, runtime);
  assert.equal(r.library_sha256, runtime['librotquant_ggml_test.so']);
  for (const c of [...r.cases, ...r.decode_cases]) {
    assert.equal(c.passed, true);
    assert.ok(Number.isFinite(c.max_abs) && c.max_abs >= 0);
  }
}
assert.equal(candidateOps.cases.length, 42);
assert.equal(candidateOps.decode_cases.length, 54);
for (const c of candidateOps.decode_cases) assert.equal(c.exact_reference, true);

function checkLimits(metrics, limits) {
  for (const key of ['max_abs_error', 'mean_abs_error', 'mean_kl']) {
    assert.ok(Number.isFinite(metrics[key]) && metrics[key] >= 0 && metrics[key] <= limits[key], key);
  }
  assert.ok(metrics.top1_agreement >= (limits.top1_agreement_min ?? limits.top1_agreement));
}
for (const prefix of ['', 'decode4-']) for (const bits of [6, 8]) {
  const r = stage(`${prefix}whole-model-w${bits}`);
  assert.equal(r.passed, true);
  assert.equal(r.cpu_fallback_forbidden, true);
  assert.ok(r.gpu_custom_ops > 0);
  assert.deepEqual(r.runtime_files, runtime);
  assert.ok(Object.values(r.guards).every(x => x === true));
  checkLimits(r.metrics, r.thresholds);
  for (const m of Object.values(r.by_prompt_length)) checkLimits(m, r.thresholds);
  if (prefix) assert.equal(r.model_sha256, stage(`whole-model-w${bits}`).model_sha256);
}
const conversion = stage('hf-conversion');
assert.equal(conversion.passed, true);
assert.deepEqual(conversion.runtime_files, runtime);
for (const c of conversion.cases) {
  assert.equal(c.passed, true);
  checkLimits(c.metrics, conversion.thresholds);
}
const retained = stage('retained-b5_v6_s0', 'retained/report.json');
const candidateRetained = stage('decode4-retained-b5_v6_s0', 'retained/report.json');
const exported = stage('export-b5_v6_s0', 'export.json');
const source = stage('source-b5_v6_s0', 'source.json');
for (const r of [retained, candidateRetained]) {
  assert.equal(r.passed, true);
  assert.deepEqual(r.runtime_files, runtime);
  assert.equal(r.cpu_fallback_forbidden, true);
  assert.ok(r.gpu_custom_ops > 0);
  assert.equal(r.export_sha256, filehash(stagepath('export-b5_v6_s0', 'export.json')));
  assert.equal(r.probe_sha256, source.probe_sha256);
  assert.equal(r.checkpoint_manifest_sha256, source.checkpoint_sha256);
  assert.deepEqual(r.export_files, exported.artifact_files);
  near(r.complete_model_payload_bytes, sum(Object.values(r.export_files).map(x => x.bytes)));
  checkLimits(r.parity, r.parity.thresholds);
  assert.ok(Object.values(r.parity.guards).every(x => x === true));
  assert.equal(r.parity.positions, 16);
  assert.equal(r.parity.exact_generation_probes, 4);
  assert.equal(r.parity.generation_probes, 4);
}
// Equality of summary metrics, not a new bitwise full-model output comparison.
assert.deepEqual(candidateRetained.parity, retained.parity);

const timings = [];
for (const context of [128, 512]) {
  const referencePath = stagepath(`pilot-b5_v6_s0-ctx${context}`, 'pilot/report.json');
  const ref = read(referencePath);
  for (const label of ['pilot', 'decode4']) {
    const relative = stagepath(`${label}-b5_v6_s0-ctx${context}`, 'pilot/report.json');
    const r = read(relative), candidate = label === 'decode4';
    assert.equal(r.passed, true);
    assert.equal(r.measurement_kind, 'throughput');
    assert.equal(r.rows.length, 4);
    assert.equal(r.summary.measured_repetitions, 3);
    assert.deepEqual(r.runtime_files, runtime);
    assert.equal(r.settings.rq3_kernel, candidate ? 'decode4' : 'reference');
    assert.deepEqual({...r.settings, rq3_kernel: 'reference'}, ref.settings);
    assert.equal(r.settings.rq3_profile, false);
    assert.equal(r.settings.cuda_graphs_disabled, false);
    assert.equal(r.cpu_fallback_forbidden, true);
    for (const key of ['context', 'context_capacity', 'controls', 'prompt_ids_sha256', 'backend',
      'export_sha256', 'probe_sha256', 'complete_model_payload_bytes']) assert.deepEqual(r[key], ref[key]);
    assert.equal(r.parity_report_sha256, filehash(stagepath(
      `${candidate ? 'decode4-' : ''}retained-b5_v6_s0`, 'retained/report.json')));
    for (const [i, row] of r.rows.entries()) {
      assert.equal(row.completed, true);
      assert.equal(row.warmup, i === 0);
      assert.equal(row.input_tokens, context);
      assert.equal(row.decode_steps, 32);
      assert.equal(row.decode_step_seconds.length, 32);
      assert.ok(row.prefill_seconds > 0 && row.decode_step_seconds.every(x => x > 0));
      assert.deepEqual(row.decode_input_ids, ref.rows[1].decode_input_ids);
      assert.equal(row.decode_input_ids.length, 32);
      near(row.decode_seconds, sum(row.decode_step_seconds));
      near(row.prefill_tokens_per_second, context / row.prefill_seconds);
      near(row.decode_tokens_per_second, 32 / row.decode_seconds);
      if (candidate) {
        assert.equal(row.prefill_profile.instrumented, false);
        assert.equal(row.decode_profile.instrumented, false);
        assert.ok(row.prefill_profile.tiled_host_dispatches > 0);
        assert.ok(row.decode_profile.decode_host_dispatches > 0);
        // Graph replay need not increase host counters once per GPU invocation.
      }
    }
    const measured = r.rows.slice(1);
    near(r.summary.prefill_tokens_per_second, context * 3 / sum(measured.map(x => x.prefill_seconds)));
    near(r.summary.decode_tokens_per_second, 96 / sum(measured.map(x => x.decode_seconds)));
    near(r.summary.decode_tps_min, Math.min(...measured.map(x => x.decode_tokens_per_second)));
    near(r.summary.decode_tps_max, Math.max(...measured.map(x => x.decode_tokens_per_second)));
    if (candidate) {
      checkBinding(r.native_diagnostics.binding);
      assert.equal(r.replay_report_sha256, filehash(referencePath));
      const c = comparison.comparisons.find(x => x.context === context && x.label === label);
      assert.equal(c.reference_report_sha256, filehash(referencePath));
      assert.equal(c.candidate_report_sha256, filehash(relative));
      near(c.prefill_rate_ratio_to_reference, r.summary.prefill_tokens_per_second / ref.summary.prefill_tokens_per_second);
      near(c.decode_rate_ratio_to_reference, r.summary.decode_tokens_per_second / ref.summary.decode_tokens_per_second);
      near(c.sampled_vram_ratio_to_reference, r.memory.sampled_peak_process_vram_mib / ref.memory.sampled_peak_process_vram_mib);
    }
    timings.push({label, context, ...r.summary, sampled_peak_mib: r.memory.sampled_peak_process_vram_mib});
  }
}
const result = JSON.stringify({revision, checks: {included_artifact_references: included,
  external_artifacts_absent: absent.size, producer_files: Object.keys(code).length,
  candidate_operator_cases: candidateOps.cases.length, exact_reference_decode_cases: candidateOps.decode_cases.length},
  passed_stages: 20, active_minutes: summary.active_minutes, runtime_build_revision: build.root_revision,
  timings, comparisons: comparison.comparisons, retained_parity: candidateRetained.parity,
  complete_model_payload_bytes: retained.complete_model_payload_bytes,
  boundary: 'Offline receipt/hash/arithmetic and reported-guard audit. Binaries, model tensors and raw parity logits are absent; no GPU replay, new quality measurement, or automatic promotion.'}, null, 2) + '\n';
if (process.argv[3]) fs.writeFileSync(process.argv[3], result, {flag: 'wx'});
else process.stdout.write(result);
