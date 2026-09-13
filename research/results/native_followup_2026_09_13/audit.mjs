// Offline receipt/measurement audit. Does not execute a model or validate CUDA.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import cp from 'node:child_process';
import assert from 'node:assert/strict';
const root = process.argv[2];
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const filehash = p => hash(fs.readFileSync(path.join(root, p)));
const stagepath = (name, file='report.json') => `stages/${name}/attempt-001/${file}`;
const stage = (name, file) => read(stagepath(name, file));
const sum = xs => xs.reduce((a,b)=>a+b,0);
const near = (a,b) => assert.ok(Number.isFinite(a) && Math.abs(a-b) <= 1e-8*Math.max(1,Math.abs(b)), `${a} != ${b}`);
const workflow = read('workflow.json'), summary = read('summary.json');
const comparison = read('comparison-b5_v6_s0.json');
const revision = workflow.controls.revision;
assert.equal(revision, '304d3297743abad675d7696f799dc07e9c20bdae');
assert.equal(workflow.controls.dirty_local_test, false);
assert.equal(summary.status, 'failed');
assert.equal(summary.stages.length, 23);
for (const [i,row] of summary.stages.entries()) {
  assert.equal(row.status, i<22 ? 'passed' : 'failed');
  assert.equal(row.name, workflow.attempts[i].name);
  near(row.elapsed_seconds, workflow.attempts[i].elapsed_seconds);
}
assert.equal(summary.stages.at(-1).name, 'decode4-operators-CUDA0');
near(sum(summary.stages.map(x=>x.elapsed_seconds)), workflow.active_seconds);
near(workflow.active_seconds/60, summary.active_minutes);
const remoteRoot = workflow.attempts[0].directory.split('/stages/')[0];
let included=0; const absent=new Set();
for (const attempt of workflow.attempts) for (const [name, expected] of Object.entries(attempt.artifacts || {})) {
  if (name.startsWith(remoteRoot+'/')) {
    const relative=name.slice(remoteRoot.length+1);
    assert.equal(filehash(relative), expected.sha256);
    assert.equal(fs.statSync(path.join(root, relative)).size, expected.bytes);
    included++;
  } else absent.add(name);
}
const runtime=stage('binding-load','load.json').runtime_files;
const build=stage('native-build-and-load','build-receipt.json');
assert.equal(stage('native-build-and-load','cache.json').cache_hit, true);
assert.equal(runtime['librotquant_ggml_test.so'],build.library_sha256);
const code={...workflow.controls.workflow_sources,...build.external_sources};
for (const [name,sha] of Object.entries(code)) assert.equal(hash(cp.execFileSync('git',['show',`${revision}:${name}`])),sha,name);
for (const kind of ['bf16','q4_0']) {
  const name=`conventional-preflight-${kind}`, r=stage(name);
  assert.equal(r.passed,true); assert.equal(r.gpu_executed,true);
  assert.deepEqual(r.runtime_files,runtime);
  const relative=stagepath(name,'conventional-probes/probes.json');
  assert.equal(filehash(relative),r.probes_sha256);
  const captures=read(relative).captures;
  for (const device of ['CPU','CUDA0']) {
    assert.deepEqual(captures.private_bridge[device],captures.public_api[device]);
    assert.equal(r.same_backend[device].passed,true);
    assert.equal(r.same_backend[device].metrics.max_abs_error,0);
  }
  assert.equal(r.cross_backend.private_bridge.passed,kind==='bf16');
  assert.equal(r.cross_backend.public_api.passed,kind==='bf16');
}
const timings=[];
for (const context of [128,512]) {
  const referencePath=stagepath(`pilot-b5_v6_s0-ctx${context}`,'pilot/report.json');
  const ref=read(referencePath);
  for (const label of ['pilot','bf16','ud_q4']) {
    const relative=stagepath(`${label}-b5_v6_s0-ctx${context}`,'pilot/report.json'), r=read(relative);
    assert.equal(r.passed,true); assert.equal(r.measurement_kind,'throughput');
    assert.equal(r.rows.length,4); assert.equal(r.summary.measured_repetitions,3);
    assert.deepEqual(r.runtime_files,runtime); assert.deepEqual(r.settings,ref.settings);
    assert.equal(r.settings.rq3_profile,false); assert.equal(r.settings.cuda_graphs_disabled,false);
    for (const key of ['context','context_capacity','controls','prompt_ids_sha256','backend']) assert.deepEqual(r[key],ref[key]);
    for (const [i,row] of r.rows.entries()) {
      assert.equal(row.completed,true); assert.equal(row.warmup,i===0);
      assert.equal(row.decode_steps,32); assert.equal(row.input_tokens,context);
      assert.equal(row.decode_step_seconds.length,32);
      assert.ok(row.prefill_seconds>0 && row.decode_step_seconds.every(x=>x>0));
      near(row.decode_seconds,sum(row.decode_step_seconds));
      near(row.prefill_tokens_per_second,context/row.prefill_seconds);
      near(row.decode_tokens_per_second,32/row.decode_seconds);
      assert.deepEqual(row.decode_input_ids,ref.rows[1].decode_input_ids);
    }
    near(r.summary.prefill_tokens_per_second,context*3/sum(r.rows.slice(1).map(x=>x.prefill_seconds)));
    near(r.summary.decode_tokens_per_second,96/sum(r.rows.slice(1).map(x=>x.decode_seconds)));
    if (label!=='pilot') {
      assert.equal(r.reference_report_sha256,filehash(referencePath));
      assert.equal(r.baseline_receipt_sha256,filehash(stagepath(`prepare-${label}`,'baseline.json')));
      const c=comparison.comparisons.find(x=>x.label===label && x.context===context);
      assert.equal(c.reference_report_sha256,filehash(referencePath));
      assert.equal(c.candidate_report_sha256,filehash(relative));
      near(c.prefill_rate_ratio_to_reference,r.summary.prefill_tokens_per_second/ref.summary.prefill_tokens_per_second);
      near(c.decode_rate_ratio_to_reference,r.summary.decode_tokens_per_second/ref.summary.decode_tokens_per_second);
      near(c.sampled_vram_ratio_to_reference,r.memory.sampled_peak_process_vram_mib/ref.memory.sampled_peak_process_vram_mib);
    }
    timings.push({label,context,...r.summary,sampled_peak_mib:r.memory.sampled_peak_process_vram_mib});
  }
}
assert.equal(comparison.completed,false); assert.equal(comparison.comparisons.length,4);
assert.ok(fs.readFileSync(path.join(root,stagepath('decode4-operators-CUDA0','logs/check_rq3_gpu.log')),'utf8').includes('Candidate decode/prefill dispatch missing'));
const result=JSON.stringify({revision,checks:{included_artifacts:included,external_artifacts_absent:absent.size,producer_files:Object.keys(code).length},
  passed_stages:22,failed_stages:1,active_minutes:summary.active_minutes,timings,
  boundary:'Receipt/hash/arithmetic verification only. No GPU replay, candidate promotion or new quality measurement.'},null,2)+'\n';
if(process.argv[3]) fs.writeFileSync(process.argv[3],result,{flag:'wx'});
else console.log(result);
