"""Render a standalone browser report from measured benchmark episodes."""
from __future__ import annotations

import json
from pathlib import Path

from plotly.offline import get_plotlyjs


def render_report(payload: dict, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    data = data.replace("<", r"\u003c")
    html = _HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", data)
    output.write_text(html, encoding="utf-8")
    return output


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Explore measured scheduling policies, nurse routes, and appointment outcomes in a home health care simulation.">
<title>Home care dispatch benchmark</title>
<style>
:root{color-scheme:light;--ink:#152d35;--muted:#4d6269;--teal:#08776c;--green:#165d51;--warm:#985220;--line:#d7e2df;--page:#f3f7f5;--white:#fff;--wash:#e8f2ef}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:1rem/1.55 system-ui,-apple-system,sans-serif}
main,.wrap{max-width:1300px;margin:auto;padding-inline:clamp(1rem,4vw,3rem)}
header{background:#123831;color:#fff;border-bottom:4px solid #58b8a0}
header .wrap{padding-block:clamp(2rem,6vw,4.5rem)}
header p{max-width:750px;color:#dbefea;margin:.8rem 0 1.2rem}
h1,h2,h3{line-height:1.14;margin:0;font-weight:720;letter-spacing:-.035em}
h1{font-size:clamp(2rem,6vw,4.1rem)}
h2{font-size:clamp(1.65rem,3vw,2.35rem)}
h3{font-size:1.2rem}
a{color:var(--green);text-underline-offset:.2em}
header a{color:#ddf7ef}
nav{display:flex;flex-wrap:wrap;gap:.6rem 1.5rem}
section{margin-block:clamp(2rem,4.5vw,3.6rem);scroll-margin-top:1rem}
section>p{max-width:800px;margin:.6rem 0 1rem;color:var(--muted)}
.facts,.toolbar,.cards{display:flex;flex-wrap:wrap;gap:.7rem}
.facts{margin-block:1.35rem}
.facts span{padding:.35rem .65rem;background:#ffffff16;border:1px solid #ffffff40;border-radius:.38rem;font-size:.9rem}
.toolbar{align-items:end;gap:.8rem 1rem;margin:1rem 0}
label{display:grid;gap:.25rem;font-size:.875rem;font-weight:680;min-width:10rem}
input,select{font:inherit;color:var(--ink);padding:.46rem .7rem;border:1px solid #a4bcb6;border-radius:.4rem;background:#fff;max-width:100%;min-height:2.55rem}
select{appearance:none;padding-right:2.2rem;background-image:linear-gradient(45deg,transparent 50%,var(--green) 50%),linear-gradient(135deg,var(--green) 50%,transparent 50%);background-position:calc(100% - 1.05rem) 53%,calc(100% - .75rem) 53%;background-size:.31rem .31rem;background-repeat:no-repeat}
:focus-visible{outline:3px solid #167b6d;outline-offset:3px}
.panel,.stat{min-width:0;background:var(--white);border:1px solid var(--line);border-radius:.65rem}
.panel{overflow:hidden;margin-bottom:1rem}
.panel>h3,.panel>p,.panel>.toolbar{margin:.9rem 1rem}
.plot{width:100%;min-height:320px}
.split{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}
.cards{margin:0 0 1rem}
.stat{flex:1 1 130px;padding:.7rem .9rem}
.stat span{display:block;font-size:.875rem;color:var(--muted)}
.stat strong{display:block;font-size:1.28rem;font-variant-numeric:tabular-nums}
.hint{color:var(--muted);font-size:.9rem}
.scroll{overflow:auto;max-height:27rem}
.scroll.compact{max-height:none}
table{border-collapse:collapse;width:100%;text-align:left;font-size:.875rem;line-height:1.4;font-variant-numeric:tabular-nums}
td,th{padding:.56rem .72rem;white-space:nowrap;border-bottom:1px solid #e5edeb}
thead{background:#eaf3ef;color:#173c34;position:sticky;z-index:1;top:0}
tbody tr:nth-child(even){background:#f7faf9}
tbody tr[data-active="true"]{background:#d9eee7}
.status{font-weight:700}
.status[data-status="Completed"]{color:#146451}
.status[data-status="Window missed"],.status[data-status="Unassigned"],.status[data-status="Interrupted"]{color:#894510}
.link-button{background:none;border:0;padding:0;color:var(--green);font:inherit;font-weight:700;text-decoration:underline;text-underline-offset:.15em;cursor:pointer}
.methodology{padding:1.05rem 1.15rem}
.methodology p{max-width:870px}
.methodology ul{padding-left:1.25rem}
details{padding:.7rem 0;border-top:1px solid var(--line)}
summary{font-weight:700;cursor:pointer}
code{white-space:pre-wrap;overflow-wrap:anywhere}
footer{border-top:1px solid var(--line);color:var(--muted);padding-block:1.4rem;margin-top:2rem}
@media(max-width:820px){.split{grid-template-columns:1fr}}
@media(max-width:460px){.toolbar label{min-width:0;width:100%}.plot{min-height:280px}}
</style>
</head>
<body>
<header><div class="wrap">
<h1>Home care dispatch benchmark</h1>
<p>Follow how scheduling policies assign nurses to home visits under skill, appointment-window, and shift constraints. Compare outcomes across simulation seeds and inspect individual shifts.</p>
<div class="facts" id="facts"></div>
<nav aria-label="Report sections">
  <a href="#comparison">Policy comparison</a><a href="#episodes">Episode explorer</a><a href="#methodology">Methods and tuning</a>
  <a href="#training" id="training-link" hidden>Training diagnostics</a>
</nav>
</div></header>
<main>
<section id="comparison">
<h2>Policy comparison</h2>
<p id="comparison-note"></p>
<div class="toolbar">
  <label>Metric <select id="metric"></select></label>
  <label>Evaluation seed <select id="comparison-seed"></select></label>
</div>
<div class="panel"><div class="plot" id="comparison-chart" aria-label="Policy comparison chart"></div></div>
<div class="panel scroll compact"><table aria-label="Policy comparison values">
<thead><tr><th scope="col">Policy</th><th scope="col">Visits completed</th><th scope="col">Completion</th><th scope="col">Reward</th><th scope="col">Travel min</th><th scope="col">Wait min</th><th scope="col">Service min</th><th scope="col">Shift min</th><th scope="col">Decisions</th><th scope="col">Runtime s</th><th scope="col">Truncation</th></tr></thead>
<tbody id="comparison-rows"></tbody></table></div>
</section>
<section id="episodes">
<h2>Episode explorer</h2>
<p>Choose a policy and seed to replay its recorded decisions. Route markers give visit order. Select a job on either chart or in the allocation table to inspect its events.</p>
<div class="toolbar">
  <label>Policy <select id="policy"></select></label>
  <label>Evaluation seed <select id="episode-seed"></select></label>
  <label>Nurse <select id="nurse"></select></label>
  <label>Appointment <select id="job"></select></label>
</div>
<div class="cards" id="episode-metrics" aria-live="polite"></div>
<div class="split">
  <div class="panel">
    <h3>Routes and visit order</h3>
    <p class="hint">Plan view in dataset coordinates. Dotted legs show intended destinations when travel is interrupted.</p>
    <div class="plot" id="routes" aria-label="Nurse routes"></div>
  </div>
  <div class="panel">
    <h3>Nurse schedules</h3>
    <p class="hint">Time advances from shift start. Faded segments end at an interruption or at the recorded shift boundary.</p>
    <div class="plot" id="timeline" aria-label="Nurse schedules"></div>
  </div>
</div>
<div class="panel">
  <h3>Job allocation</h3>
  <p class="hint">Window bounds apply to the start of service. Route order counts arrivals for each nurse. Times are simulated minutes.</p>
  <div class="toolbar"><label>Outcome <select id="job-status"></select></label></div>
  <p class="hint" id="job-count" aria-live="polite"></p>
  <div class="scroll"><table aria-label="Job allocation">
  <thead><tr><th scope="col">Job</th><th scope="col">Elder</th><th scope="col">Skill</th><th scope="col">Nurse</th><th scope="col">Route order</th><th scope="col">Window</th><th scope="col">Dispatch</th><th scope="col">Arrival</th><th scope="col">Service start</th><th scope="col">Service end</th><th scope="col">Outcome</th></tr></thead>
  <tbody id="job-rows"></tbody></table></div>
</div>
<div class="panel">
  <h3>Event log</h3>
  <div class="toolbar">
    <label>Event type <select id="event-type"></select></label>
    <label>Search events <input id="event-query" type="search" placeholder="Type, nurse, job, or detail"></label>
  </div>
  <p class="hint" id="event-count" aria-live="polite"></p>
  <div class="scroll"><table aria-label="Event log">
  <thead><tr><th scope="col">Time</th><th scope="col">Event</th><th scope="col">Nurse</th><th scope="col">Job</th><th scope="col">Detail</th></tr></thead>
  <tbody id="event-rows"></tbody></table></div>
</div>
</section>
<section id="methodology">
<h2>Methods and tuning</h2>
<p>See the scheduling rules, training setup, and validation results used to choose the reinforcement-learning model.</p>
<div class="panel methodology">
<h3>Experiment setup</h3>
<ul id="setup"></ul>
<h3>Dispatch policies</h3>
<ul id="methods"></ul>
<h3>Optuna trials</h3>
<p class="hint">Trial and final-training scores are validation rewards. Open each algorithm to see the training budget and parameters.</p>
<div id="tuning"></div>
</div>
</section>
<section id="training" hidden>
<h2 id="training-title">Training diagnostics</h2>
<p id="training-note"></p>
<div class="toolbar">
  <label>Agent <select id="training-agent"></select></label>
  <label>Metric <select id="training-metric"></select></label>
</div>
<div class="panel"><div class="plot" id="training-chart" aria-label="Final training diagnostics"></div></div>
</section>
<footer>Research demonstrator. Route geometry uses the dataset distance model. Clinical use would require additional validation.</footer>
</main>
<noscript>Enable JavaScript to explore the embedded results.</noscript>
<script>__PLOTLY__</script>
<script type="application/json" id="benchmark-data">__DATA__</script>
<script>
"use strict";
const {meta, scenario, methods, tuning, runs} = JSON.parse(document.getElementById("benchmark-data").textContent);
const $ = id => document.getElementById(id);
const byMethod = Object.fromEntries(methods.map(method => [method.id, method]));
const seeds = [...new Set(runs.map(run => run.seed))];
const nurseIds = scenario.nurse_skills.map((_, nurse) => nurse);
const runIndex = new Map(runs.map(run => [run.method + "/" + run.seed, run]));
const metrics = {
  jobs_completed: ["Visits completed", 1], completion_rate: ["Completion (%)", 1, 100],
  jobs_unserved: ["Visits unserved", 1], total_reward: ["Reward", 1],
  travel_minutes: ["Travel (nurse-minutes)", 1], waiting_minutes: ["Waiting (nurse-minutes)", 1],
  service_minutes: ["Service (nurse-minutes)", 1], time: ["Shift elapsed (minutes)", 1],
  decisions: ["Dispatch decisions", 1], wall_seconds: ["Runtime (seconds)", 3],
  truncated: ["Shift truncation (%)", 1, 100]
};
const palette = ["#08776c","#2852a0","#a6456b","#9e5b12","#7352a1","#227790","#415d3a","#ae4434","#446071","#885a33"];
const activityColors = {Travel:"#50677d", Waiting:"#b56c18", Service:"#08776c", Return:"#3464a8"};
const statuses = ["Completed", "Window missed", "Interrupted", "Unassigned"];
const plotConfig = {responsive:true, displaylogo:false, scrollZoom:false};
const plotLayout = {
  paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"system-ui, sans-serif",color:"#152d35",size:12},
  margin:{t:24,r:28,b:54,l:62}, hoverlabel:{align:"left",namelength:-1}
};

function node(tag, text = "") {
  const el = document.createElement(tag);
  el.textContent = text;
  return el;
}
function options(id, pairs, preferred) {
  const el = $(id);
  const value = preferred ?? el.value;
  el.replaceChildren(...pairs.map(([key, label]) => new Option(label, key)));
  if (pairs.some(([key]) => String(key) === String(value))) el.value = value;
}
function row(values) {
  const tr = node("tr");
  for (const value of values) {
    const cell = node("td");
    value instanceof Node ? cell.append(value) : cell.textContent = value;
    tr.append(cell);
  }
  return tr;
}
function linkButton(label, callback) {
  const button = node("button", label);
  button.className = "link-button";
  button.type = "button";
  button.onclick = callback;
  return button;
}
const isNumber = v => v !== null && v !== undefined && Number.isFinite(Number(v));
const fmt = (v, decimals = 1) => isNumber(v) ? Number(v).toFixed(decimals) : "—";
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const stdev = (values, center) => Math.sqrt(values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1));
const nurseLabel = n => "Nurse " + n;
const titleCase = type => type.replaceAll("_", " ").replace(/^./, letter => letter.toUpperCase());
const metricValue = (run, key) => Number(run.metrics[key]) * (metrics[key][2] ?? 1);
const groupName = method => method.group === "rl" ? "RL" : "Baseline";
const bestMethod = byMethod[meta.best_rl];
const methodLabel = method => method.label + (method.id === meta.best_rl ? " (selected RL)" : "");

function comparison() {
  const key = $("metric").value;
  const allSeeds = $("comparison-seed").value === "";
  const chosenSeed = Number($("comparison-seed").value);
  $("comparison-note").textContent = (allSeeds
    ? "Bars show means across held-out seeds. Open circles mark the sample standard deviation in both directions when two or more seeds are available. Dots represent individual episodes."
    : "Each row shows the recorded outcome for the selected held-out seed.") +
    (bestMethod ? " " + bestMethod.label + " was selected on validation reward." : "");
  const comparisons = methods.map(method => {
    const sample = allSeeds ? seeds.map(seed => runIndex.get(method.id + "/" + seed)).filter(Boolean)
      : [runIndex.get(method.id + "/" + chosenSeed)].filter(Boolean);
    const values = sample.map(run => metricValue(run, key)).filter(Number.isFinite);
    const average = values.length ? mean(values) : null;
    return {method, sample, average, sd:values.length > 1 ? stdev(values, average) : null};
  });
  const labels = methods.map(method => method.label);
  const bars = {
    type:"bar", x:labels, y:comparisons.map(item => item.average),
    marker:{color:methods.map(method => method.id === meta.best_rl ? "#985220" : method.group === "rl" ? "#2852a0" : "#08776c")},
    customdata:comparisons.map(item => [groupName(item.method),item.sample.length]),
    hovertemplate:"%{x} · %{customdata[0]}<br>%{y:.3f}<br>%{customdata[1]} episode(s)<extra></extra>"
  };
  const traces = [bars];
  if (allSeeds) {
    const eligible = comparisons.filter(item => item.sd !== null);
    const endpoints = eligible.flatMap(item => [item.average - item.sd,item.average + item.sd]);
    traces.push({
      type:"scatter", mode:"markers", x:eligible.flatMap(item => [item.method.label,item.method.label]),
      y:endpoints, marker:{symbol:"circle-open",size:9,color:"#183d38",line:{width:1.6}},
      hovertemplate:"%{x}<br>Mean ± sample SD endpoint: %{y:.3f}<extra></extra>"
    });
    const points = comparisons.flatMap(item => item.sample.map(run => [item.method.label,metricValue(run,key),run.seed]));
    traces.push({type:"scatter",mode:"markers",x:points.map(p=>p[0]),y:points.map(p=>p[1]),customdata:points.map(p=>p[2]),
      marker:{size:7,color:"#fff",line:{color:"#183d38",width:1.6}}, hovertemplate:"%{x}<br>Seed %{customdata}: %{y:.3f}<extra></extra>"});
  }
  Plotly.react($("comparison-chart"),traces,{
    ...plotLayout, height:350,showlegend:false,bargap:.37,
    xaxis:{categoryorder:"array",categoryarray:labels,automargin:true},
    yaxis:{title:{text:metrics[key][0]},gridcolor:"#e5eeec",automargin:true,zeroline:true}
  },plotConfig);
  const columns = ["jobs_completed","completion_rate","total_reward","travel_minutes","waiting_minutes","service_minutes","time","decisions","wall_seconds","truncated"];
  $("comparison-rows").replaceChildren(...comparisons.map(({method,sample}) => {
    const button=linkButton(methodLabel(method),() => {
      $("policy").value=method.id;
      $("nurse").value="";
      $("job").value="";
      refreshEpisode(allSeeds ? undefined : chosenSeed);
      $("episodes").scrollIntoView();
    });
    const values=columns.map(col => {
      const numbers=sample.map(run=>metricValue(run,col)).filter(Number.isFinite);
      const decimals=allSeeds ? metrics[col][1] : ["jobs_completed","decisions","truncated"].includes(col) ? 0 : metrics[col][1];
      return numbers.length ? fmt(mean(numbers),decimals)+(metrics[col][2] ? "%" : "") : "—";
    });
    const tr=row([button,...values]);
    tr.dataset.active=method.id===meta.best_rl;
    return tr;
  }));
}

function readEpisode(run) {
  const jobs = new Map(scenario.jobs.map(job => [job.id, {...job, status:"Unassigned"}]));
  const routes = nurseIds.map(() => [{point:scenario.depot, label:"Depot departure", job:null}]);
  const counts = nurseIds.map(() => 0);
  const active = new Map();
  const segments = [];
  const partialRoutes = [];
  const start = (activity, event) => active.set(event.nurse, {...event, activity});
  const stop = (event, partial = false) => {
    const first = active.get(event.nurse);
    if (!first) return;
    segments.push({activity:first.activity,nurse:first.nurse,job:first.job,start:first.time,end:event.time,partial});
    if (partial && ["Travel","Return"].includes(first.activity) && isNumber(first.x1)) {
      partialRoutes.push({nurse:first.nurse,from:[first.x0,first.y0],to:[first.x1,first.y1],job:first.job});
    }
    active.delete(event.nurse);
  };
  for (const event of run.events) {
    const job=jobs.get(event.job);
    const route=routes[event.nurse];
    switch (event.type) {
      case "dispatch":
        Object.assign(job,{nurse:event.nurse,dispatched:event.time,status:"Interrupted"});
        break;
      case "travel_start": start("Travel",event); break;
      case "arrive":
        stop(event);
        job.arrival=event.time;
        job.order=++counts[event.nurse];
        route.push({point:job.position,label:"Visit "+job.order+" · Job "+job.id+" · Arrival "+fmt(event.time)+" min",job:job.id,order:job.order});
        break;
      case "wait_start": start("Waiting",event); break;
      case "service_start":
        stop(event);
        start("Service",event);
        job.service_start=event.time;
        break;
      case "service_end":
        stop(event);
        job.service_end=event.time;
        job.status="Completed";
        break;
      case "missed_window":
        stop(event);
        job.status="Window missed";
        break;
      case "return_start": start("Return",event); break;
      case "return_end":
        if (route.length>1) route.push({point:scenario.depot,label:"Depot return · "+fmt(event.time)+" min",job:null});
        stop(event);
        break;
    }
  }
  const finalTime = run.events.at(-1)?.type === "end" ? run.events.at(-1).time : run.metrics.time;
  for (const nurse of active.keys()) stop({nurse,time:finalTime},true);
  return {run,jobs,routes,segments,partialRoutes};
}

let episode;
const nurseFilter = nurse => $("nurse").value === "" || nurse === Number($("nurse").value);
const jobFilter = job => $("job").value === "" || job === Number($("job").value);
function focusJob(id) {
  if (id === null || id === undefined) return;
  $("job").value=String(id);
  $("job-status").value="";
  $("event-type").value="";
  $("event-query").value="";
  updateFocus();
}
async function drawChart(id,traces,layout) {
  const chart=$(id);
  await Plotly.react(chart,traces,layout,plotConfig);
  if (!chart.dataset.jobClick) {
    chart.dataset.jobClick="bound";
    chart.on("plotly_click", ({points}) => {
      const selected=points[0].customdata;
      focusJob(Array.isArray(selected) ? selected[0] : selected);
    });
  }
}
function routeChart() {
  const matching=[...episode.jobs.values()].filter(job=>nurseFilter(job.nurse));
  const coordinates=matching.map(job=>job.position);
  const depot=scenario.depot;
  const traces=[
    {type:"scatter",mode:"markers",name:"Appointments",x:coordinates.map(p=>p[0]),y:coordinates.map(p=>p[1]),
      customdata:matching.map(j=>j.id),text:matching.map(j=>"Job "+j.id+" · Elder "+j.elder_id+"<br>Floor coordinate "+j.position[2]+" · "+j.status+"<br>Window "+fmt(j.window_start,0)+" to "+fmt(j.window_end,0)+" min"),
      hovertemplate:"%{text}<extra></extra>",
      marker:{size:matching.map(job=>$("job").value!==""&&jobFilter(job.id)?17:8),color:matching.map(job=>job.status==="Completed"?"#9eb9b2":"#bc7327"),line:{color:"#fff",width:1}}},
    {type:"scatter",mode:"markers",name:"Depot",x:[depot[0]],y:[depot[1]],
      marker:{symbol:"diamond",size:15,color:"#172f35"},hovertemplate:"Depot<extra></extra>"}
  ];
  for (const nurse of nurseIds.filter(nurseFilter)) {
    const route=episode.routes[nurse];
    if (route.length<2) continue;
    const color=palette[nurse%palette.length];
    traces.push({type:"scatter",mode:"lines+markers+text",name:nurseLabel(nurse),
      x:route.map(visit=>visit.point[0]),y:route.map(visit=>visit.point[1]),
      text:route.map(visit=>visit.order ?? ""),textposition:"top center",textfont:{color,size:11},
      customdata:route.map(visit=>[visit.job,visit.label]),hovertemplate:"%{customdata[1]}<extra>%{fullData.name}</extra>",
      line:{color,width:2},marker:{color,size:route.map(visit=>visit.job!==null&&$("job").value!==""&&jobFilter(visit.job)?13:7)}
    });
  }
  for (const leg of episode.partialRoutes.filter(leg=>nurseFilter(leg.nurse))) {
    traces.push({type:"scatter",mode:"lines+markers",name:nurseLabel(leg.nurse),
      x:[leg.from[0],leg.to[0]],y:[leg.from[1],leg.to[1]],customdata:[leg.job,leg.job],
      text:[nurseLabel(leg.nurse),leg.job!==null?"Target job "+leg.job:"Target depot"],
      hovertemplate:"%{text}<br>Interrupted journey<extra></extra>",
      line:{color:palette[leg.nurse%palette.length],dash:"dot",width:2},marker:{symbol:"circle-open",size:9},showlegend:false
    });
  }
  drawChart("routes",traces,{
    ...plotLayout,height:410,legend:{orientation:"h",y:-.18},margin:{t:15,r:20,b:80,l:56},uirevision:episode.run.method+"/"+episode.run.seed+"/"+$("nurse").value,
    xaxis:{title:{text:"X coordinate"},gridcolor:"#e5eeec",zeroline:false},
    yaxis:{title:{text:"Y coordinate"},gridcolor:"#e5eeec",zeroline:false,scaleanchor:"x",scaleratio:1}
  });
}
function timelineChart() {
  const visibleNurses=nurseIds.filter(nurseFilter);
  const traces=Object.entries(activityColors).map(([activity,color]) => {
    const items=episode.segments.filter(segment=>segment.activity===activity&&nurseFilter(segment.nurse));
    return {type:"bar",orientation:"h",name:activity,y:items.map(segment=>nurseLabel(segment.nurse)),
      x:items.map(segment=>Math.max(0,segment.end-segment.start)),base:items.map(segment=>segment.start),
      customdata:items.map(segment=>[segment.job,segment.job!==null?"Job "+segment.job:"Depot",segment.end,segment.partial?" · interrupted":""]),
      hovertemplate:"%{y} · %{customdata[1]}<br>%{base:.2f} to %{customdata[2]:.2f} min<br>%{x:.2f} min%{customdata[3]}<extra>%{fullData.name}</extra>",
      marker:{color,opacity:items.map(segment=>segment.partial?.45:$("job").value===""||jobFilter(segment.job)?1:.26),
        line:{color:"#152d35",width:items.map(segment=>$("job").value!==""&&jobFilter(segment.job)?2:0)}}
    };
  });
  drawChart("timeline",traces,{
    ...plotLayout,height:410,barmode:"overlay",legend:{orientation:"h",y:-.18},margin:{t:15,r:20,b:80,l:85},uirevision:episode.run.method+"/"+episode.run.seed+"/"+$("nurse").value,
    xaxis:{title:{text:"Simulated minutes"},range:[0,meta.horizon],gridcolor:"#e5eeec"},
    yaxis:{categoryorder:"array",categoryarray:visibleNurses.map(nurseLabel).reverse(),type:"category",tickmode:"array",tickvals:visibleNurses.map(nurseLabel)}
  });
}
function jobTable() {
  const status=$("job-status").value;
  const visible=[...episode.jobs.values()].filter(job=>nurseFilter(job.nurse)&&jobFilter(job.id)&&(!status||job.status===status));
  $("job-count").textContent=visible.length+" of "+scenario.jobs.length+" appointments";
  $("job-rows").replaceChildren(...visible.map(job=>{
    const badge=node("span",job.status);
    badge.className="status";
    badge.dataset.status=job.status;
    const tr=row([
      linkButton(String(job.id),()=>focusJob(job.id)),job.elder_id,job.demand,job.nurse ?? "—",job.order ?? "—",
      fmt(job.window_start,0)+"–"+fmt(job.window_end,0),fmt(job.dispatched),fmt(job.arrival),fmt(job.service_start),fmt(job.service_end),badge
    ]);
    tr.dataset.active=$("job").value!==""&&jobFilter(job.id);
    return tr;
  }));
}
function eventTable() {
  const query=$("event-query").value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const type=$("event-type").value;
  const source=episode.run.events;
  const visible=source.filter(event=>{
    const content=[titleCase(event.type),event.type,event.nurse!==null?"nurse "+event.nurse:"",event.job!==null?"job "+event.job:"",event.detail].join(" ").toLowerCase();
    return nurseFilter(event.nurse)&&jobFilter(event.job)&&(!type||event.type===type)&&query.every(term=>content.includes(term));
  });
  $("event-count").textContent=visible.length+" of "+source.length+" events";
  $("event-rows").replaceChildren(...visible.map(event=>{
    const detail=event.detail||(isNumber(event.duration)?"Sampled service "+fmt(event.duration,2)+" min":"—");
    return row([fmt(event.time,2),titleCase(event.type),event.nurse ?? "—",event.job ?? "—",detail]);
  }));
}
function updateFocus() {
  routeChart();
  timelineChart();
  jobTable();
  eventTable();
}
function refreshEpisode(preferredSeed) {
  const available=runs.filter(run=>run.method===$("policy").value);
  options("episode-seed",available.map(run=>[run.seed,run.seed]),preferredSeed);
  const run=runIndex.get($("policy").value+"/"+$("episode-seed").value);
  episode=readEpisode(run);
  const selected=[
    ["Visits completed",run.metrics.jobs_completed+" / "+run.metrics.jobs_total],
    ["Reward",fmt(run.metrics.total_reward)],["Elapsed shift",fmt(run.metrics.time)+" min"],
    ["Travel",fmt(run.metrics.travel_minutes)+" min"],["Waiting",fmt(run.metrics.waiting_minutes)+" min"],
    ["Service",fmt(run.metrics.service_minutes)+" min"],["Shift ended",run.metrics.truncated?"At time limit":"Normally"]
  ];
  $("episode-metrics").replaceChildren(...selected.map(([label,value])=>{
    const stat=node("div"); stat.className="stat"; stat.append(node("span",label),node("strong",value)); return stat;
  }));
  const types=[...new Set(run.events.map(event=>event.type))].sort();
  options("event-type",[["","All events"],...types.map(type=>[type,titleCase(type)])]);
  updateFocus();
}

function methodology() {
  $("facts").replaceChildren(...[
    "Instance "+meta.instance,meta.jobs+" appointments",meta.nurses+" nurses",methods.length+" policies",seeds.length+" test seeds"
  ].map(value=>node("span",value)));
  const trainingSeeds = meta.training_seeds ?? [meta.training_seed].filter(isNumber);
  const usesCuda = !meta.devices || Object.values(meta.devices).includes("cuda");
  const widths = meta.network_label ?? (meta.network_search?.length
    ? "Optuna searched hidden widths of "+meta.network_search.join(", ")+" units per layer. Selected widths appear with each agent's parameters."
    : "Hidden layer widths "+(meta.network ?? [64,64]).join(" × ")+".");
  const notes=[
    "Each action assigns an eligible appointment to the current idle nurse or defers the nurse. Nurses travel, wait, and serve patients concurrently.",
    "The dispatch mask checks nurse skill, the start window, waiting limits, and forecast return to depot. Actual service times are uncertain, so the shift can end mid-activity.",
    "Reward adds 10 per completed visit and subtracts 8 per unresolved visit, 0.03 per travel minute, 0.05 per waiting minute, and 3 per invalid action. Nurse-minute totals count completed activity intervals.",
    meta.action_encoding,meta.observation_normalization,meta.service_randomness,meta.selection,
    "Training seeds "+trainingSeeds.join(", ")+". "+(meta.validation_seeds.length
      ? "Validation seeds "+meta.validation_seeds.join(", ")+". " : "")+"Held-out test seeds "+meta.test_seeds.join(", ")+".",
    widths,
    meta.parameters_from
      ? "Each RL algorithm used the parameters from "+meta.parameters_from+" and "+meta.final_steps+" requested steps per training seed."
      : "Each RL algorithm ran "+meta.trials+" Optuna trials with "+meta.trial_steps+" requested steps per trial and "+meta.final_steps+" requested final steps per training seed.",
    "Training device "+meta.device+(usesCuda && meta.gpu ? " · "+meta.gpu : "")+(usesCuda && meta.cuda ? " · PyTorch CUDA "+meta.cuda : "")+". Benchmark wall time "+fmt(meta.runtime_seconds,1)+" seconds.",
    meta.versions && "Package versions "+Object.entries(meta.versions).map(([pkg,version])=>pkg+" "+version).join(", ")+"."
  ].filter(Boolean);
  $("setup").replaceChildren(...notes.map(value=>node("li",value)));
  $("methods").replaceChildren(...methods.map(method=>{
    const item=node("li");
    item.append(node("strong",method.label+" ("+groupName(method)+") "),document.createTextNode(method.description));
    return item;
  }));
  $("tuning").replaceChildren(...Object.entries(tuning).map(([id,trialSet])=>{
    const detail=node("details");
    detail.open=id===meta.best_rl;
    const overview=methodLabel(byMethod[id])+" · Final validation reward "+fmt(trialSet.final_validation_reward,2);
    const bestTrial=trialSet.trials.length ? "Best trial validation reward "+fmt(trialSet.best_value,2)+". " : "";
    const replicates=trialSet.replicates ?? [];
    const budget=bestTrial+(replicates.length
      ? "Final training used "+replicates.length+" independent seeds."
      : "Final training consumed "+trialSet.final_training_steps+" steps in "+fmt(trialSet.final_training_seconds,2)+" seconds.");
    const params=node("p"); params.append(node("strong","Selected parameters "),node("code",JSON.stringify(trialSet.best_params)));
    const scroller=node("div"); scroller.className="scroll";
    const table=node("table"); table.setAttribute("aria-label",byMethod[id].label+" Optuna trials");
    const head=table.createTHead().insertRow();
    head.append(...["Trial","Validation reward","Training steps","Seconds","Parameters"].map(label=>{const th=node("th",label); th.scope="col"; return th;}));
    table.createTBody().append(...trialSet.trials.map(trial=>row([trial.number,fmt(trial.value,2),trial.steps,fmt(trial.seconds,2),node("code",JSON.stringify(trial.params))])));
    scroller.append(table);
    detail.append(node("summary",overview),node("p",budget),params);
    if (trialSet.trials.length) detail.append(scroller);
    if (replicates.length) {
      const runsScroller=node("div"); runsScroller.className="scroll";
      const runsTable=node("table"); runsTable.setAttribute("aria-label",byMethod[id].label+" final training runs");
      const runsHead=runsTable.createTHead().insertRow();
      runsHead.append(...["Training seed","Steps","Completed episodes","Seconds","Validation reward"].map(label=>{const th=node("th",label); th.scope="col"; return th;}));
      runsTable.createTBody().append(...replicates.map(run=>row([
        run.seed===trialSet.representative_seed ? run.seed+" (held-out checkpoint)" : run.seed,
        run.steps,run.episodes,fmt(run.seconds,2),fmt(run.validation_reward,2)
      ])));
      runsScroller.append(runsTable);
      detail.append(node("h4","Final training runs"),runsScroller);
    }
    return detail;
  }));
}

function trainingDiagnostics() {
  const labels = {
    "train/loss":"Training loss", "train/value_loss":"Value loss",
    "train/policy_gradient_loss":"Policy-gradient loss", "train/entropy_loss":"Entropy loss",
    "train/approx_kl":"Approximate KL divergence", "rollout/ep_rew_mean":"Training episode reward (rolling mean)"
  };
  const series = new Map(methods.map(method => [
    method.id, Object.entries(labels).flatMap(([key,label]) => {
      const result = tuning[method.id];
      const profile = (result?.loss_profiles?.[key] ?? [])
        .filter(point => isNumber(point.steps) && isNumber(point.mean));
      if (profile.length) return [{key,label,points:profile,aggregate:true}];
      const points = (result?.training_history ?? [])
        .filter(point => isNumber(point.steps) && isNumber(point[key]));
      return points.length ? [{key,label,points}] : [];
    })
  ]).filter(([,diagnostics]) => diagnostics.length));
  if (!series.size) return;
  $("training").hidden = $("training-link").hidden = false;
  if ([...series.values()].some(diagnostics=>diagnostics.some(item=>item.aggregate))) {
    $("training-title").textContent = $("training-link").textContent = "Training loss profiles";
  }
  options("training-agent",[...series.keys()].map(id=>[id,byMethod[id].label]),meta.best_rl);
  function drawTrainingMetric() {
    const agent = $("training-agent").value;
    const {key,label,points,aggregate} = series.get(agent).find(item=>item.key===$("training-metric").value);
    const x = points.map(point=>point.steps);
    const color = agent === "dqn" ? "#2852a0" : "#08776c";
    const traces = [];
    $("training-note").textContent = (aggregate
      ? meta.loss_profile_note ?? "The line shows the mean across independent training seeds and the band shows ±1 sample standard deviation."
      : "The chart shows metrics from one final training run.") + " The agents optimize different objectives. Assessing policy convergence also requires reward and behavior to stabilize across training steps and seeds.";
    if (aggregate && points.some(point=>point.n >= 2 && isNumber(point.std))) {
      const bound = sign => points.map(point=>point.n >= 2 && isNumber(point.std) ? point.mean + sign * point.std : null);
      traces.push({type:"scatter",mode:"lines",x,y:bound(-1),line:{width:0},hoverinfo:"skip",showlegend:false});
      traces.push({type:"scatter",mode:"lines",x,y:bound(1),line:{width:0},hoverinfo:"skip",
        fill:"tonexty",fillcolor:agent === "dqn" ? "rgba(40,82,160,.19)" : "rgba(8,119,108,.19)",name:"±1 sample SD"});
    }
    traces.push({type:"scatter",mode:"lines+markers",x,
      y:points.map(point=>aggregate ? point.mean : point[key]),name:aggregate ? "Mean" : label,
      line:{color},marker:{color,size:5},
      customdata:aggregate ? points.map(point=>[fmt(point.std,5),point.n]) : undefined,
      hovertemplate:aggregate
        ? "Step %{x:,d}<br>Mean %{y:.5g}<br>Sample SD %{customdata[0]}<br>Training seeds %{customdata[1]}<extra></extra>"
        : "Step %{x:,d}<br>%{y:.5g}<extra></extra>"});
    Plotly.react($("training-chart"),traces,{
      ...plotLayout,height:350,uirevision:agent+"/"+key,
      showlegend:aggregate,legend:{orientation:"h",x:0,y:1.06},
      xaxis:{title:{text:"Environment steps"},gridcolor:"#e5eeec"},
      yaxis:{title:{text:label},gridcolor:"#e5eeec",automargin:true}
    },plotConfig);
  }
  function selectTrainingAgent() {
    options("training-metric",series.get($("training-agent").value).map(({key,label})=>[key,label]));
    drawTrainingMetric();
  }
  $("training-agent").onchange=selectTrainingAgent;
  $("training-metric").onchange=drawTrainingMetric;
  selectTrainingAgent();
}

options("metric",Object.entries(metrics).map(([id,settings])=>[id,settings[0]]),"jobs_completed");
options("comparison-seed",[["","All test seeds"],...seeds.map(seed=>[seed,seed])]);
options("policy",methods.map(method=>[method.id,methodLabel(method)]),meta.best_rl);
options("nurse",[["","All nurses"],...nurseIds.map(nurse=>[nurse,nurseLabel(nurse)+" · Skill "+scenario.nurse_skills[nurse]])]);
options("job",[["","All appointments"],...scenario.jobs.map(job=>[job.id,"Job "+job.id+" · Elder "+job.elder_id])]);
options("job-status",[["","All outcomes"],...statuses.map(status=>[status,status])]);
for (const id of ["metric","comparison-seed"]) $(id).onchange=comparison;
for (const id of ["policy","episode-seed"]) $(id).onchange=()=>refreshEpisode();
for (const id of ["nurse","job"]) $(id).onchange=updateFocus;
$("job-status").onchange=jobTable;
$("event-type").onchange=eventTable;
$("event-query").oninput=eventTable;
methodology();
trainingDiagnostics();
comparison();
refreshEpisode();
</script>
</body>
</html>
"""
