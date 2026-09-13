// Offline audit of followup4-profile. Does not load binaries or execute a model.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import cp from 'node:child_process';
import assert from 'node:assert/strict';
const root = path.resolve(process.argv[2]);
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const filehash = p => hash(fs.readFileSync(path.join(root, p)));
const location = (name, file = 'report.json') => `stages/${name}/attempt-001/${file}`;
const stage = (name, file) => read(location(name, file));
const sum = values => values.reduce((a, b) => a + b, 0);
const near = (a, b) => assert.ok(Number.isFinite(a) && Number.isFinite(b) &&
  Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(b)), `${a} != ${b}`);
const names = ['rotation', 'matrix', 'vocabulary_head', 'embedding'];
const workflow = read('workflow.json'), summary = read('summary.json');
const comparison = read('comparison-b5_v6_s0.json');
const revision = workflow.controls.revision;
assert.equal(revision, '53b8dac6cecf316c22070b2931fee800155072f8');
assert.equal(workflow.controls.dirty_local_test, false);
assert.equal(workflow.controls.diagnostic_profiles, true);
assert.deepEqual(workflow.controls.pilot_contexts, [128]);
assert.deepEqual(workflow.controls.conventional_baselines, []);
assert.equal(workflow.status, 'passed');
assert.equal(summary.status, 'passed');
assert.equal(summary.error, null);
assert.equal(summary.stages.length, 20);
assert.equal(workflow.attempts.length, 20);
assert.equal(new Set(summary.stages.map(x => x.name)).size, 20);
for (const [i, s] of summary.stages.entries()) {
  assert.equal(s.status, 'passed');
  assert.equal(s.name, workflow.attempts[i].name);
  assert.equal(workflow.attempts[i].status, 'passed');
  near(s.elapsed_seconds, workflow.attempts[i].elapsed_seconds);
}
near(sum(summary.stages.map(x => x.elapsed_seconds)), workflow.active_seconds);
near(workflow.active_seconds / 60, summary.active_minutes);
const remoteRoot = workflow.attempts[0].directory.split('/stages/')[0];
let included = 0;
const absent = new Set();
for (const a of workflow.attempts) for (const [name, expected] of Object.entries(a.artifacts || {})) {
  if (!name.startsWith(remoteRoot + '/')) { absent.add(name); continue; }
  const relative = name.slice(remoteRoot.length + 1);
  assert.ok(path.resolve(root, relative).startsWith(root + path.sep));
  assert.equal(filehash(relative), expected.sha256, relative);
  assert.equal(fs.statSync(path.join(root, relative)).size, expected.bytes);
  included++;
}
const runtime = stage('binding-load', 'load.json').runtime_files;
const build = stage('native-build-and-load', 'build-receipt.json');
assert.equal(stage('native-build-and-load', 'cache.json').cache_hit, true);
assert.equal(runtime['librotquant_ggml_test.so'], build.library_sha256);
const code = {...workflow.controls.workflow_sources, ...build.external_sources};
for (const [name, sha] of Object.entries(code)) {
  assert.equal(hash(cp.execFileSync('git', ['show', `${revision}:${name}`])), sha, name);
}
assert.equal(hash(cp.execFileSync('git', ['show', `${revision}:integrations/llama.cpp/rotquant-native-v2.patch`])), build.patch_sha256);
const candidate = stage('decode4-operators-CUDA0');
assert.equal(candidate.decode_cases.length, 54);
for (const r of candidate.decode_cases) assert.equal(r.exact_reference, true);
for (const name of ['candidate-dispatch-preflight', 'operators-CPU', 'operators-CUDA0',
  'decode4-operators-CUDA0', 'whole-model-w6', 'whole-model-w8',
  'decode4-whole-model-w6', 'decode4-whole-model-w8', 'hf-conversion']) {
  const r = stage(name);
  assert.equal(r.passed, true);
  assert.deepEqual(r.runtime_files, runtime);
  for (const c of [...(r.cases || []), ...(r.decode_cases || [])]) assert.equal(c.passed, true);
  if (r.guards) assert.ok(Object.values(r.guards).every(x => x === true));
}
const retained = stage('retained-b5_v6_s0', 'retained/report.json');
const candidateRetained = stage('decode4-retained-b5_v6_s0', 'retained/report.json');
for (const r of [retained, candidateRetained]) {
  assert.equal(r.passed, true);
  assert.equal(r.cpu_fallback_forbidden, true);
  assert.deepEqual(r.runtime_files, runtime);
  assert.equal(r.export_sha256, filehash(location('export-b5_v6_s0', 'export.json')));
  for (const key of ['max_abs_error', 'mean_abs_error', 'mean_kl']) {
    assert.ok(Number.isFinite(r.parity[key]) && r.parity[key] >= 0 && r.parity[key] <= r.parity.thresholds[key]);
  }
  assert.equal(r.parity.top1_agreement, 1);
  assert.equal(r.parity.positions, 16);
  assert.equal(r.parity.exact_generation_probes, 4);
  assert.equal(r.parity.generation_probes, 4);
  assert.ok(Object.values(r.parity.guards).every(x => x === true));
}
assert.deepEqual(candidateRetained.parity, retained.parity);
const refPath = location('pilot-b5_v6_s0-ctx128', 'pilot/report.json');
const ref = read(refPath);
const timings = [], profiles = [];
for (const [label, kind, kernel] of [['pilot', 'pilot', 'reference'], ['decode4', 'pilot', 'decode4'],
  ['profile-reference', 'profile', 'reference'], ['profile-decode4', 'profile', 'decode4']]) {
  const relative = location(`${label}-b5_v6_s0-ctx128`, `${kind}/report.json`);
  const r = read(relative), diagnostic = kind === 'profile';
  assert.equal(r.passed, true);
  assert.equal(r.status, 'passed');
  assert.equal(r.measurement_kind, diagnostic ? 'diagnostic' : 'throughput');
  assert.equal(r.settings.rq3_kernel, kernel);
  assert.equal(r.settings.rq3_profile, diagnostic);
  assert.equal(r.settings.cuda_graphs_disabled, diagnostic);
  assert.equal(r.cpu_fallback_forbidden, true);
  assert.deepEqual(r.runtime_files, runtime);
  assert.deepEqual({...r.settings, rq3_kernel: 'reference', rq3_profile: false, cuda_graphs_disabled: false}, ref.settings);
  for (const key of ['context', 'context_capacity', 'controls', 'prompt_ids_sha256', 'backend',
    'export_sha256', 'probe_sha256', 'complete_model_payload_bytes']) assert.deepEqual(r[key], ref[key]);
  assert.equal(r.parity_report_sha256, filehash(location(
    `${kernel === 'decode4' ? 'decode4-' : ''}retained-b5_v6_s0`, 'retained/report.json')));
  if (label !== 'pilot') assert.equal(r.replay_report_sha256, filehash(refPath));
  assert.equal(r.rows.length, 4);
  for (const [i, row] of r.rows.entries()) {
    assert.equal(row.completed, true);
    assert.equal(row.warmup, i === 0);
    assert.equal(row.input_tokens, 128);
    assert.equal(row.decode_steps, 32);
    assert.equal(row.decode_step_seconds.length, 32);
    assert.deepEqual(row.decode_input_ids, ref.rows[1].decode_input_ids);
    assert.equal(row.decode_input_ids.length, 32);
    assert.ok(row.prefill_seconds > 0 && row.decode_step_seconds.every(x => x > 0));
    near(row.decode_seconds, sum(row.decode_step_seconds));
    near(row.prefill_tokens_per_second, 128 / row.prefill_seconds);
    near(row.decode_tokens_per_second, 32 / row.decode_seconds);
  }
  const measured = r.rows.slice(1);
  assert.equal(r.summary.measured_repetitions, 3);
  near(r.summary.prefill_tokens_per_second, 384 / sum(measured.map(x => x.prefill_seconds)));
  near(r.summary.decode_tokens_per_second, 96 / sum(measured.map(x => x.decode_seconds)));
  if (!diagnostic) {
    timings.push({kernel, ...r.summary, sampled_peak_mib: r.memory.sampled_peak_process_vram_mib});
    continue;
  }
  const receipt = comparison.profiles.find(x => x.kernel === kernel);
  assert.equal(receipt.context, 128);
  assert.equal(receipt.report_sha256, filehash(relative));
  assert.equal(r.native_diagnostics.instrumented, true);
  const binding = r.native_diagnostics.binding;
  assert.equal(binding.lookup, 'execution-library dependency handle');
  assert.equal(path.basename(binding.provider), 'libggml-cuda.so.0');
  assert.equal(binding.provider_sha256, runtime['libggml-cuda.so.0']);
  for (const row of r.rows) for (const phase of ['prefill', 'decode']) {
    const p = row[`${phase}_profile`];
    assert.equal(p.instrumented, true);
    const multiplier = phase === 'decode' ? 32 : 1;
    for (const name of names) {
      assert.ok(Number.isFinite(p.operators[name].milliseconds) && p.operators[name].milliseconds > 0);
      assert.equal(p.operators[name].host_dispatches,
        multiplier * (['matrix', 'rotation'].includes(name) ? 200 : 1));
    }
    assert.equal(p.decode_host_dispatches, kernel === 'decode4' && phase === 'decode' ? 6400 : 0);
    assert.equal(p.tiled_host_dispatches, kernel === 'decode4' && phase === 'prefill' ? 200 : 0);
  }
  for (const name of names) {
    near(r.native_diagnostics.operators[name].milliseconds,
      sum(r.rows.flatMap(x => ['prefill', 'decode'].map(p => x[`${p}_profile`].operators[name].milliseconds))));
  }
  for (const phase of ['prefill', 'decode']) {
    const means = Object.fromEntries(names.map(n => [n, sum(measured.map(x => x[`${phase}_profile`].operators[n].milliseconds)) / 3]));
    const total = sum(Object.values(means));
    for (const n of names) profiles.push({kernel, phase, operator: n,
      mean_gpu_ms_per_repetition: means[n], share_of_custom_event_percent: 100 * means[n] / total,
      ...(phase === 'decode' ? {mean_gpu_ms_per_decode_step: means[n] / 32} : {})});
  }
}
assert.equal(comparison.completed, true);
assert.equal(comparison.comparisons.length, 1);
assert.equal(comparison.profiles.length, 2);
const c = comparison.comparisons[0];
assert.equal(c.reference_report_sha256, filehash(refPath));
assert.equal(c.candidate_report_sha256, filehash(location('decode4-b5_v6_s0-ctx128', 'pilot/report.json')));
near(c.prefill_rate_ratio_to_reference, timings[1].prefill_tokens_per_second / timings[0].prefill_tokens_per_second);
near(c.decode_rate_ratio_to_reference, timings[1].decode_tokens_per_second / timings[0].decode_tokens_per_second);
near(c.sampled_vram_ratio_to_reference, timings[1].sampled_peak_mib / timings[0].sampled_peak_mib);
const result = JSON.stringify({revision, passed_stages: 20, active_minutes: summary.active_minutes,
  checks: {included_artifact_references: included, external_artifacts_absent: absent.size, producer_files: Object.keys(code).length},
  timings, comparisons: comparison.comparisons, profiles, retained_parity: retained.parity,
  boundary: 'Offline hash/arithmetic and reported-gate audit; no GPU replay. Profiles are custom-op CUDA event time with graphs disabled, not whole-model wall shares, bandwidth or task accuracy.'}, null, 2) + '\n';
if (process.argv[3]) fs.writeFileSync(process.argv[3], result, {flag: 'wx'});
else process.stdout.write(result);
