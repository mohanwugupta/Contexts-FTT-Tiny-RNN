"use strict";

const allowedKeys = ["a", "s", "d", "f"];
const layouts = {
  A: [[12,35],[29,35],[47,35],[65,35],[65,68],[47,68],[29,68],[12,68]],
  B: [[35,12],[35,29],[35,47],[35,65],[62,65],[62,47],[62,29],[62,12]],
  Mixed: [[12,30],[29,30],[47,30],[65,30],[62,70],[47,70],[29,70],[14,70]],
  Novel: [[12,50],[23,23],[50,12],[77,23],[88,50],[77,77],[50,88],[23,77]]
};
let config, assignment, schedule, assignmentIndex, code, storageKey;
let events = [], scheduleIndex = 0, position = 0, attempt = 1, onsetMonotonic = 0, accepting = false;

const $ = id => document.getElementById(id);
function now(eventType, extra={}) {
  const row = {
    schema_version:"human-transfer-events-1.0.0", experiment:config.experiment,
    participant_uid:`${config.experiment}:${code}`, counterbalance_code:code,
    assignment_index:assignmentIndex, data_origin:$("synthetic-mode").checked ? "synthetic_task_validation" : "empirical_browser_task",
    event_type:eventType, absolute_event_order:events.length+1,
    monotonic_time_ms:performance.now(), wall_clock_time_ms:Date.now(),
    schedule_index:scheduleIndex, sequence_id:schedule?.[scheduleIndex]?.sequence_id ?? null,
    attempt, transition_position:position, ...assignment, ...(schedule?.[scheduleIndex] || {}), ...extra
  };
  delete row.target_sequence;
  events.push(row); persist(); return row;
}
function persist() {
  if (!storageKey) return;
  localStorage.setItem(storageKey, JSON.stringify({events,scheduleIndex,position,attempt,config,assignment,schedule,assignmentIndex,code}));
}
async function hashIndex(value, length) {
  const data = new TextEncoder().encode(value.trim());
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
  let number = 0n; for (let i=0;i<8;i++) number=(number<<8n)+BigInt(hash[i]);
  return Number(number % BigInt(length));
}
function emit(type, payload={}) { now(type,payload); }
function renderLayout(id) {
  const container=$("layout"); container.replaceChildren();
  (layouts[id] || layouts.Novel).forEach(([x,y]) => {
    const dot=document.createElement("span"); dot.className="layout-dot";
    dot.style.left=`${x}%`; dot.style.top=`${y}%`; container.appendChild(dot);
  });
}
function renderTargets(row) {
  const container=$("targets"); container.replaceChildren();
  row.target_sequence.forEach((token,index) => {
    const target=document.createElement("div"); target.className="target";
    if (index<position) { target.classList.add("done"); target.textContent="✓"; }
    else if (index===position) { target.classList.add("current"); target.textContent=row.response_keys[token].toUpperCase(); }
    else if (row.visibility==="full_preview") { target.textContent=row.response_keys[token].toUpperCase(); }
    else { target.classList.add("masked"); target.textContent="•"; }
    // No token/action/value is stored in attributes. Future one-target labels are absent from the DOM.
    container.appendChild(target);
  });
}
function renderSequence() {
  if (scheduleIndex>=schedule.length) return finish();
  const row=schedule[scheduleIndex]; accepting=false;
  $("phase").textContent=`${row.phase.toUpperCase()} · ${row.context}`;
  $("progress").textContent=`Sequence ${scheduleIndex+1} / ${schedule.length}`;
  $("context-cue").textContent=`Cue: ${row.cue_id}`; renderLayout(row.layout_id); renderTargets(row);
  $("feedback").textContent="";
  emit("context_onset", {cue_id:row.cue_id,layout_id:row.layout_id});
  requestAnimationFrame(frameTime => {
    onsetMonotonic=performance.now(); accepting=true;
    emit("sequence_onset", {stimulus_frame_time_ms:frameTime, stimulus_onset_monotonic_ms:onsetMonotonic});
    if ($("synthetic-mode").checked) setTimeout(simulateResponse, 12);
  });
}
function simulateResponse() {
  if (!accepting) return; const row=schedule[scheduleIndex], token=row.target_sequence[position];
  handleResponse(row.response_keys[token], false); if (scheduleIndex<schedule.length) setTimeout(simulateResponse, 12);
}
function handleResponse(key, repeated) {
  if (!accepting || !allowedKeys.includes(key)) return;
  const row=schedule[scheduleIndex], expected=row.response_keys[row.target_sequence[position]], correct=key===expected;
  const responseTime=performance.now();
  emit("response", {key_identity:key, repeated_key_suppressed:repeated, correct,
    response_interval_ms:responseTime-onsetMonotonic, response_monotonic_ms:responseTime});
  if (!correct) {
    accepting=false; emit("error", {expected_key:expected, observed_key:key, restart_policy:"current_sequence"});
    $("feedback").textContent="Incorrect — this sequence will restart."; attempt+=1; position=0;
    setTimeout(renderSequence, 500); return;
  }
  position+=1; onsetMonotonic=responseTime;
  if (position===row.target_sequence.length) {
    accepting=false; emit("sequence_end", {completed:true, scored_execution_intervals:7});
    const previousBlock=row.block_id; scheduleIndex+=1; position=0; attempt=1;
    if (scheduleIndex<schedule.length && schedule[scheduleIndex].block_id!==previousBlock) emit("block_boundary", {next_block_id:schedule[scheduleIndex].block_id});
    emit("rest", {scheduled_duration_ms:row.scheduled_pause_ms}); setTimeout(renderSequence,row.scheduled_pause_ms);
  } else renderTargets(row);
}
async function start() {
  code=$("participant-code").value.trim(); if (!code) { $("setup-error").textContent="A counterbalance code is required."; return; }
  const url=$("configuration").value;
  try { config=await fetch(url,{cache:"no-store"}).then(response => { if(!response.ok) throw new Error(response.statusText); return response.json(); }); }
  catch(error) { $("setup-error").textContent=`Could not load configuration: ${error.message}`; return; }
  assignmentIndex=await hashIndex(code,config.cells.length); assignment=config.cells[assignmentIndex]; schedule=config.schedules[assignmentIndex];
  storageKey=`human-transfer:${config.experiment}:${code}`;
  const saved=localStorage.getItem(storageKey);
  if(saved) { const state=JSON.parse(saved); events=state.events; scheduleIndex=state.scheduleIndex; position=state.position; attempt=state.attempt; }
  $("setup").classList.add("hidden"); $("task").classList.remove("hidden"); emit("task_start", {resumed:Boolean(saved)}); renderSequence();
}
function finish() { emit("task_complete"); $("task").classList.add("hidden"); $("complete").classList.remove("hidden"); }
function download(extension) {
  let text,mime;
  if(extension==="json") { text=JSON.stringify(events,null,2); mime="application/json"; }
  else { const columns=[...new Set(events.flatMap(Object.keys))]; const quote=v=>`"${String(v??"").replaceAll('"','""')}"`; text=[columns.join(","),...events.map(row=>columns.map(c=>quote(row[c])).join(","))].join("\n"); mime="text/csv"; }
  const link=document.createElement("a"); link.href=URL.createObjectURL(new Blob([text],{type:mime})); link.download=`${config.experiment}_${code}_events.${extension}`; link.click(); URL.revokeObjectURL(link.href);
}
$("start").addEventListener("click",start);
$("abort").addEventListener("click",()=>{accepting=false;emit("abort");finish();});
$("download-json").addEventListener("click",()=>download("json")); $("download-csv").addEventListener("click",()=>download("csv"));
window.addEventListener("keydown",event=>{ if(event.repeat){emit("response",{key_identity:event.key.toLowerCase(),repeated_key_suppressed:true,correct:false,excluded:true});return;} handleResponse(event.key.toLowerCase(),false); });
window.addEventListener("blur",()=>{ if(config) emit("blur",{accepting_at_blur:accepting}); accepting=false; });
window.addEventListener("focus",()=>{ if(config){emit("focus"); if(scheduleIndex<schedule.length)renderSequence();} });
window.addEventListener("beforeunload",()=>{ if(config && scheduleIndex<schedule.length) emit("abort",{reason:"beforeunload"}); });
