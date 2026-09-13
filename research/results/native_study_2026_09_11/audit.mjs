import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import cp from 'node:child_process';
import assert from 'node:assert/strict';

const root = process.argv[2];
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const filehash = p => hash(fs.readFileSync(path.join(root, p)));
const stagepath = (s, n='report.json') => `stages/${s}/attempt-001/${n}`;
const stage = (s,n) => read(stagepath(s,n));
const sum = xs => xs.reduce((a,b)=>a+b,0);
const near = (a,b) => assert.ok(Number.isFinite(a) && Math.abs(a-b) <= 1e-8 * Math.max(1,Math.abs(b)), `${a} != ${b}`);
const workflow = read('workflow.json'), summary = read('summary.json');
const comparison = read('comparison-b5_v6_s0.json');
const revision = workflow.controls.revision;
assert.equal(revision, 'ed920464932a6eab09bda263fb547efa3e364b74');
assert.equal(workflow.controls.dirty_local_test, false);
assert.equal(summary.status, 'failed');
assert.equal(summary.stages.length, 25);
for(let i=0;i<25;i++) {
  assert.equal(summary.stages[i].status, i<24 ? 'passed':'failed');
  assert.equal(summary.stages[i].name, workflow.attempts[i].name);
  near(summary.stages[i].elapsed_seconds, workflow.attempts[i].elapsed_seconds);
}
near(sum(summary.stages.map(s=>s.elapsed_seconds)), workflow.active_seconds);
near(workflow.active_seconds/60,summary.active_minutes);
const remoteRoot = workflow.attempts[0].directory.split('/stages/')[0];
let artifacts=0; const absent=new Set();
for (const a of workflow.attempts) for(const [name, expected] of Object.entries(a.artifacts || {})) {
  if(name.startsWith(remoteRoot+'/')) {
    const relative=name.slice(remoteRoot.length+1);
    assert.equal(filehash(relative),expected.sha256);
    assert.equal(fs.statSync(path.join(root,relative)).size,expected.bytes);
    artifacts++;
  } else absent.add(name);
}
const build=stage('native-build-and-load','build-receipt.json');
const runtime=stage('binding-load','load.json').runtime_files;
assert.equal(runtime['librotquant_ggml_test.so'],build.library_sha256);
const code={...workflow.controls.workflow_sources,...build.external_sources};
for(const [name,sha] of Object.entries(code)) assert.equal(hash(cp.execFileSync('git',['show',`${revision}:${name}`])),sha,name);
const timings=[];
for(const context of [128,512,2048]) {
  const pair=[];
  for(const variant of ['pilot','tiled4']) {
    const relative=stagepath(`${variant}-b5_v6_s0-ctx${context}`,'pilot/report.json');
    const r=read(relative);
    assert.equal(r.passed,true); assert.equal(r.measurement_kind,'throughput');
    assert.equal(r.settings.rq3_profile,false); assert.equal(r.settings.cuda_graphs_disabled,false);
    assert.deepEqual(r.runtime_files,runtime); assert.equal(r.rows.length,4);
    const parityName=variant==='pilot'?'retained-b5_v6_s0':'tiled4-retained-b5_v6_s0';
    assert.equal(r.parity_report_sha256,filehash(stagepath(parityName,'retained/report.json')));
    const gate=stage(parityName,'retained/report.json');
    assert.equal(r.export_sha256,gate.export_sha256); assert.equal(r.probe_sha256,gate.probe_sha256);
    for(const [i,row] of r.rows.entries()) {
      assert.equal(row.completed,true); assert.equal(row.warmup,i===0);
      assert.equal(row.decode_steps,32); assert.equal(row.input_tokens,context);
      assert.equal(row.decode_step_seconds.length,32);
      assert.ok(row.prefill_seconds>0 && row.decode_step_seconds.every(x=>x>0));
      near(row.decode_seconds,sum(row.decode_step_seconds));
      near(row.prefill_tokens_per_second,context/row.prefill_seconds);
      near(row.decode_tokens_per_second,32/row.decode_seconds);
      assert.deepEqual(row.decode_input_ids,r.rows[1].decode_input_ids);
    }
    const measured=r.rows.slice(1);
    near(r.summary.prefill_tokens_per_second,context*3/sum(measured.map(x=>x.prefill_seconds)));
    near(r.summary.decode_tokens_per_second,96/sum(measured.map(x=>x.decode_seconds)));
    assert.equal(r.summary.measured_repetitions,3);
    pair.push(r);
  }
  const [a,b]=pair;
  for(const key of ['prompt_ids_sha256','context_capacity','controls','probe_sha256','export_sha256']) assert.deepEqual(a[key],b[key]);
  const sa={...a.settings},sb={...b.settings}; delete sa.rq3_kernel; delete sb.rq3_kernel;
  assert.deepEqual(sa,sb); assert.deepEqual(a.rows[1].decode_input_ids,b.rows[1].decode_input_ids);
  assert.equal(b.replay_report_sha256,filehash(stagepath(`pilot-b5_v6_s0-ctx${context}`,'pilot/report.json')));
  assert.ok(b.native_diagnostics.tiled_host_dispatches>0);
  const c=comparison.comparisons.find(x=>x.context===context);
  assert.equal(c.reference_report_sha256,filehash(stagepath(`pilot-b5_v6_s0-ctx${context}`,'pilot/report.json')));
  assert.equal(c.candidate_report_sha256,filehash(stagepath(`tiled4-b5_v6_s0-ctx${context}`,'pilot/report.json')));
  near(c.prefill_rate_ratio_to_reference,b.summary.prefill_tokens_per_second/a.summary.prefill_tokens_per_second);
  near(c.decode_rate_ratio_to_reference,b.summary.decode_tokens_per_second/a.summary.decode_tokens_per_second);
  timings.push({context,old_prefill:a.summary.prefill_tokens_per_second,new_prefill:b.summary.prefill_tokens_per_second,
    prefill_ratio:c.prefill_rate_ratio_to_reference,old_decode:a.summary.decode_tokens_per_second,new_decode:b.summary.decode_tokens_per_second,
    old_peak_mib:a.memory.sampled_peak_process_vram_mib,new_peak_mib:b.memory.sampled_peak_process_vram_mib,
    old_prefill_seconds:sum(a.rows.slice(1).map(x=>x.prefill_seconds))/3,new_prefill_seconds:sum(b.rows.slice(1).map(x=>x.prefill_seconds))/3});
}
const profiles=[];
for(const variant of ['reference','tiled4']) {
  const relative=stagepath(`profile-${variant}-b5_v6_s0-ctx128`,'profile/report.json'), r=read(relative);
  assert.equal(r.passed,true); assert.equal(r.measurement_kind,'diagnostic');
  assert.equal(r.settings.rq3_profile,true); assert.equal(r.settings.cuda_graphs_disabled,true);
  assert.equal(r.rows.length,4); assert.deepEqual(r.runtime_files,runtime);
  assert.equal(filehash(relative),comparison.profiles.find(x=>x.kernel===variant).report_sha256);
  const measured=r.rows.filter(x=>!x.warmup && x.completed); assert.equal(measured.length,3);
  for(const phase of ['prefill','decode']) {
    const operators={};
    for(const operator of ['rotation','matrix','vocabulary_head','embedding']) {
      const values=measured.map(x=>x[phase+'_profile'].operators[operator]);
      assert.ok(values.every(x=>Number.isFinite(x.milliseconds)&&x.milliseconds>=0&&x.host_dispatches>0));
      operators[operator]={mean_ms:sum(values.map(x=>x.milliseconds))/3,mean_dispatches:sum(values.map(x=>x.host_dispatches))/3};
    }
    const wallms=sum(measured.map(x=>x[phase+'_seconds']))/3*1000;
    profiles.push({variant,phase,operators,wall_ms_per_rep:wallms,custom_ms:sum(Object.values(operators).map(x=>x.mean_ms)),
      head_share_custom:operators.vocabulary_head.mean_ms/sum(Object.values(operators).map(x=>x.mean_ms)),
      head_ms_per_token:phase==='decode'?operators.vocabulary_head.mean_ms/32:null});
  }
}
const parity=[];
for(const name of ['retained-b5_v6_s0','tiled4-retained-b5_v6_s0']) {
  const r=stage(name,'retained/report.json');
  assert.equal(r.passed,true); assert.deepEqual(r.runtime_files,runtime); assert.ok(Object.values(r.parity.guards).every(Boolean));
  for(const k of ['max_abs_error','mean_abs_error','mean_kl']) assert.ok(r.parity[k]<=r.parity.thresholds[k]);
  assert.equal(r.parity.top1_agreement,1);assert.equal(r.parity.exact_generation_probes,4);
  parity.push({name,...r.parity});
}
assert.deepEqual(parity[0].guards,parity[1].guards);
const operators=stage('tiled4-operators-CUDA0'); assert.equal(operators.cases.length,36);
assert.equal(operators.passed,true); assert.ok(operators.cases.every(x=>x.passed));
for(const n of ['operators-CPU','operators-CUDA0','whole-model-w6','whole-model-w8','hf-conversion','tiled4-whole-model-w6','tiled4-whole-model-w8']) {
 const r=stage(n); assert.equal(r.passed,true); assert.deepEqual(r.runtime_files,runtime);
 if(r.guards) assert.ok(Object.values(r.guards).every(Boolean));
}
const baseline=stage('bf16-b5_v6_s0-ctx128','pilot/report.json');
assert.equal(baseline.passed,false);assert.equal(baseline.rows.length,0);assert.equal(baseline.in_progress.phase,'prefill');
const log=fs.readFileSync(path.join(root,stagepath('bf16-b5_v6_s0-ctx128','logs/run_native_gguf_baseline.log')),'utf8');
assert.ok(log.includes('model.input_embed (GET_ROWS) scheduled on CPU'));
console.log(JSON.stringify({checks:{included_artifacts:artifacts,external_artifacts_absent:absent.size,producer_files:Object.keys(code).length},
 revision,passed_stages:24,failed_stages:1,active_minutes:summary.active_minutes,build_minutes:summary.stages[2].elapsed_seconds/60,
 timings,profiles,parity},null,2));
