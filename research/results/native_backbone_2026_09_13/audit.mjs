import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';

// Offline receipt audit; never executes uploaded code or loads absent binaries.
const root = path.resolve(process.argv[2]);
const repo = path.resolve(process.argv[3]);
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const read = p => JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const digest = p => hash(fs.readFileSync(path.join(root,p)));
const sum = x => x.reduce((a,b)=>a+b,0);
const close = (a,b) => assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a-b)<=1e-9*Math.max(1,Math.abs(b)), `${a} != ${b}`);
const w=read('workflow.json'), summary=read('summary.json'), shortlist=read('kernel-shortlist.json');
assert.equal(w.status,'passed'); assert.equal(summary.status,'passed');
assert.equal(w.attempts.length,29); assert.equal(new Set(w.attempts.map(x=>x.name)).size,29);
assert.ok(w.attempts.every(x=>x.status==='passed'));
assert.deepEqual(summary.stages,w.attempts.map(({name,status,elapsed_seconds})=>({name,status,elapsed_seconds})));
close(sum(w.attempts.map(x=>x.elapsed_seconds)),w.active_seconds);
close(w.active_seconds/60,summary.active_minutes);
assert.match(w.controls.revision,/^[a-f0-9]{40}$/);
const origin=w.attempts[0].directory.split('/stages/')[0]+'/';
const verified=new Set(), unavailable=new Set();
for(const a of w.attempts) for(const [p,v] of Object.entries(a.artifacts)) {
  if(!p.startsWith(origin)){unavailable.add(p);continue;}
  const rel=p.slice(origin.length); assert.ok(!rel.split('/').includes('..'));
  const local=path.join(root,rel);assert.ok(fs.existsSync(local),`Missing included receipt ${rel}`);
  assert.ok(!fs.lstatSync(local).isSymbolicLink());
  assert.equal(fs.statSync(local).size,v.bytes);assert.equal(digest(rel),v.sha256);verified.add(rel);
}
for(const [p,sha] of Object.entries(w.controls.workflow_sources)) {
  assert.match(p,/^[A-Za-z0-9_./-]+$/);assert.ok(!p.split('/').includes('..'));
  assert.equal(hash(execFileSync('git',['show',`${w.controls.revision}:${p}`],{cwd:repo})),sha);
}
const runtime=shortlist.runtime_files;
const screenRows=[];
for(const item of shortlist.candidates) {
  const rel=`stages/screen-${item.kernel}/attempt-001/report.json`, r=read(rel);
  assert.equal(digest(rel),item.report_sha256);assert.equal(r.passed,true);
  assert.deepEqual(r.runtime_files,runtime);
  assert.deepEqual(r.settings,{graphs_disabled:true,event_profiling:false,repetitions:3,iterations:10,rounds:2});
  assert.equal(r.rows.length,12);
  const expected=new Set();for(const [n,k] of [[2560,2560],[9216,2560],[2560,9216]])for(const t of [1,128])for(const round of [0,1])expected.add([n,k,t,round].join(','));
  const ratios={1:[],128:[]};
  for(const row of r.rows) {
    assert.ok(expected.delete([row.rows,row.width,row.tokens,row.round].join(',')));
    assert.equal(row.exact_decode4,true);
    assert.deepEqual(row.order,row.round===0?['decode4',item.kernel]:[item.kernel,'decode4']);
    for(const key of ['decode4_seconds','candidate_seconds']){assert.equal(row[key].length,3);assert.ok(row[key].every(x=>Number.isFinite(x)&&x>0));}
    ratios[row.tokens].push(sum(row.decode4_seconds)/sum(row.candidate_seconds));
    assert.equal(row.binding.provider_sha256,runtime[path.basename(row.binding.provider)]);
    for(const [kernel,d] of Object.entries(row.dispatch)) {
      assert.equal(d.instrumented,false);assert.equal(d.operators.matrix.host_dispatches,31);
      const tile=kernel==='w5s8'?'4':kernel.endsWith('tile8')?'8':kernel.endsWith('tile16')?'16':null;
      for(const t of ['4','8','16'])assert.equal(d.w5_host_dispatches[t],t===tile?31:0);
    }
  }
  assert.equal(expected.size,0);
  const means=Object.fromEntries(Object.entries(ratios).map(([k,v])=>[k==='1'?'decode':'prefill',Math.exp(sum(v.map(Math.log))/v.length)]));
  const worst=Math.min(...ratios[1],...ratios[128]);
  for(const d of [r.decision,item]) {close(d.worst_case_ratio,worst);for(const p of ['decode','prefill'])close(d.geomean_ratios[p],means[p]);assert.equal(d.eligible,worst>=.95&&Math.max(...Object.values(means))>=1.05);}
  screenRows.push({kernel:item.kernel,eligible:item.eligible,worst_ratio:worst,...means});
}
assert.deepEqual(shortlist.finalists,screenRows.filter(x=>x.eligible).sort((a,b)=>Math.max(b.decode,b.prefill)-Math.max(a.decode,a.prefill)).slice(0,2).map(x=>x.kernel));
const configurations=[['decode4','pilot-b5_v6_s0-ctx128','retained-b5_v6_s0'],...['w5s8-tile8','w5s8-tile16'].map(k=>[k,`${k}-b5_v6_s0-ctx128`,`${k}-retained-b5_v6_s0`])];
const pilots=[], retained=[], timings=[];
for(const [kernel,stage,parityStage] of configurations) {
  const rel=`stages/${stage}/attempt-001/pilot/report.json`,r=read(rel);
  const parityRel=`stages/${parityStage}/attempt-001/retained/report.json`,p=read(parityRel);
  assert.equal(r.passed,true);assert.equal(p.passed,true);assert.equal(r.parity_report_sha256,digest(parityRel));
  assert.equal(r.measurement_kind,'throughput');assert.equal(r.cpu_fallback_forbidden,true);assert.equal(r.backend,'CUDA0');
  assert.equal(r.context,128);assert.equal(r.settings.rq3_kernel,kernel);assert.equal(r.settings.rq3_profile,false);assert.equal(r.settings.cuda_graphs_disabled,false);
  assert.deepEqual(r.runtime_files,runtime);assert.deepEqual(p.runtime_files,runtime);
  assert.equal(r.native_diagnostics.instrumented,false);assert.equal(r.native_diagnostics.binding.provider_sha256,runtime[path.basename(r.native_diagnostics.binding.provider)]);
  assert.ok(r.gpu_custom_ops>0);assert.equal(r.rows.length,4);assert.equal(r.rows.filter(x=>x.warmup).length,1);
  const rows=r.rows.filter(x=>!x.warmup);assert.equal(rows.length,3);
  for(const row of r.rows) {
    assert.equal(row.completed,true);assert.equal(row.input_tokens,128);assert.equal(row.decode_steps,32);assert.equal(row.decode_input_ids.length,32);
    assert.equal(row.decode_step_seconds.length,32);assert.ok(row.decode_step_seconds.every(x=>Number.isFinite(x)&&x>0));
    close(sum(row.decode_step_seconds),row.decode_seconds);
    assert.deepEqual(row.decode_input_ids,(pilots[0]||r).rows[0].decode_input_ids);
  }
  const prefill=sum(rows.map(x=>x.input_tokens))/sum(rows.map(x=>x.prefill_seconds));
  const decode=sum(rows.map(x=>x.decode_steps))/sum(rows.map(x=>x.decode_seconds));
  close(prefill,r.summary.prefill_tokens_per_second);close(decode,r.summary.decode_tokens_per_second);
  const q=p.parity;assert.equal(q.passed,true);assert.ok(Object.values(q.guards).every(x=>x===true));
  for(const k of ['max_abs_error','mean_abs_error','mean_kl'])assert.ok(q[k]<=q.thresholds[k]);
  assert.ok(q.top1_agreement>=q.thresholds.top1_agreement_min);assert.equal(q.exact_generation_probes,q.generation_probes);
  assert.equal(q.positions,16);assert.equal(q.generation_probes,4);
  if(pilots.length) {
    const b=pilots[0]; for(const key of ['controls','export_sha256','probe_sha256','prompt_ids_sha256','complete_model_payload_bytes'])assert.deepEqual(r[key],b[key]);
    assert.deepEqual(q,retained[0].parity);assert.deepEqual(p.export_files,retained[0].export_files);
    assert.equal(r.replay_kernel,'decode4');assert.equal(r.replay_report_sha256,digest('stages/pilot-b5_v6_s0-ctx128/attempt-001/pilot/report.json'));
    const c=read(`comparison-b5_v6_s0-${kernel}.json`);assert.equal(c.completed,true);assert.equal(c.screen_sha256,digest('kernel-shortlist.json'));assert.equal(c.baseline_kernel,'decode4');
    assert.equal(c.comparisons[0].candidate_report_sha256,digest(rel));assert.equal(c.comparisons[0].reference_report_sha256,r.replay_report_sha256);
    close(prefill/timings[0].prefill_tps,c.comparisons[0].prefill_rate_ratio_to_reference);close(decode/timings[0].decode_tps,c.comparisons[0].decode_rate_ratio_to_reference);
  }
  const tile=kernel==='decode4'?null:kernel.endsWith('tile8')?'8':'16';
  for(const t of ['4','8','16'])assert.equal(r.native_diagnostics.w5_host_dispatches[t],t===tile?2400:0);
  pilots.push(r);retained.push(p);timings.push({kernel,prefill_tps:prefill,decode_tps:decode,peak_mib:r.memory.sampled_peak_process_vram_mib,repetitions:rows.length,prefill_seconds:rows.map(x=>x.prefill_seconds),decode_seconds:rows.map(x=>x.decode_seconds)});
}
const gates=[];
for(const a of w.attempts.filter(x=>/operators-|whole-model|hf-conversion/.test(x.name))) {
  const r=read(`stages/${a.name}/attempt-001/report.json`);assert.equal(r.passed,true);assert.deepEqual(r.runtime_files,runtime);
  for(const key of ['cases','decode_cases','prefill_format_cases'])for(const c of r[key]||[]) {assert.equal(c.passed,true);if('exact_reference'in c)assert.equal(c.exact_reference,true);}
  if(r.guards)assert.ok(Object.values(r.guards).every(x=>x===true));
  if(r.dispatch_probe){assert.equal(r.dispatch_probe.passed,true);assert.equal(r.dispatch_probe.exact_reference,true);}
  gates.push({stage:a.name,cases:r.cases?.length,decode_cases:r.decode_cases?.length,prefill_format_cases:r.prefill_format_cases?.length});
}
const output={status:'passed-with-scope-caveats',producer:w.controls.revision,stage_count:w.attempts.length,active_minutes:w.active_seconds/60,build_minutes:w.attempts.find(x=>x.name==='native-build-and-load').elapsed_seconds/60,verified_included_artifact_references:verified.size,unavailable_external_artifact_references:unavailable.size,verified_producer_source_hashes:Object.keys(w.controls.workflow_sources).length,screen:screenRows,timings,parity:retained[0].parity,gates,limitations:['Offline receipt and calculation checks, not an independent GPU replay.','Referenced external binaries, model tensors and raw parity logits are absent.','One A100 session, 128-token prompt, one retained W5/W6 arm; three measured repetitions, fixed decode replay.','Full-model configurations ran sequentially, not in randomized/counterbalanced order.','Retained parity is against the saved quantized checkpoint, not new full-precision quality evidence.','Process VRAM is sampled, not an exact peak or bandwidth measurement.','No conventional baseline was run in this bundle.']};
console.log(JSON.stringify(output,null,2));
if(process.argv[4])fs.writeFileSync(process.argv[4],JSON.stringify(output,null,2)+'\n',{flag:'wx'});
