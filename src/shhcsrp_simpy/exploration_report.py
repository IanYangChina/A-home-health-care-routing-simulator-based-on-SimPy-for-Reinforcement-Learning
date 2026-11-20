"""Render dispatch training experiments as self-contained HTML reports."""
from __future__ import annotations

import json
from pathlib import Path

from plotly.offline import get_plotlyjs


def render_exploration(payloads: list[dict], output: str | Path) -> Path:
    if not payloads:
        raise ValueError("Provide at least one exploration experiment.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payloads, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    round_name = payloads[0]["meta"].get("round")
    if round_name == "wait-and-returns":
        title = "Waiting costs and longer returns"
        description = "Compare stronger waiting penalties and longer reward estimates on home care instances C and D."
    elif round_name == "guided-long-returns":
        title = "Guided exploration with longer returns"
        description = "Compare short-wait and ant-colony guidance with matched controls on all four home care instances."
    else:
        title = "Guided exploration"
        description = "Compare exploration strategies and track which parts of the reward changed as DQN and PPO learned to dispatch nurses."
    html = _HTML.replace("__TITLE__", title).replace("__DESCRIPTION__", description)
    html = html.replace("__PLOTLY__", get_plotlyjs()).replace("__EXPERIMENTS__", data.replace("<", r"\u003c"))
    output.write_text(html, encoding="utf-8")
    return output


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="__DESCRIPTION__">
<title>__TITLE__ · Home care dispatch</title>
<style>
:root{color-scheme:light;--ink:#152d35;--muted:#4d6269;--green:#165d51;--line:#d7e2df;--page:#f3f7f5;--white:#fff;--wash:#e8f2ef}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:1rem/1.55 system-ui,-apple-system,sans-serif}
main,.wrap{max-width:1250px;margin:auto;padding-inline:clamp(1rem,4vw,3rem)}
header{background:#123831;color:#fff;border-bottom:4px solid #58b8a0}
header .wrap{padding-block:clamp(2rem,6vw,4.1rem)}
header p{max-width:720px;color:#dbefea;margin:.85rem 0 1.2rem}
header a{color:#ddf7ef}
h1,h2{margin:0;font-weight:720;line-height:1.13;letter-spacing:-.035em}
h1{font-size:clamp(2rem,6vw,3.9rem)}
h2{font-size:clamp(1.65rem,3vw,2.3rem)}
a{color:var(--green);text-underline-offset:.2em}
nav,.toolbar{display:flex;flex-wrap:wrap;gap:.75rem 1.25rem;align-items:end}
main section{margin:clamp(2rem,4.5vw,3.6rem) 0;scroll-margin-top:1rem}
main .layout-picker{margin:1.3rem 0 0;background:var(--wash);padding:1rem;border-radius:.55rem;align-items:center}
p{max-width:860px;color:var(--muted);margin:.55rem 0 1rem}
.toolbar{margin:1rem 0}
.layout-picker p{margin:0}
label{display:grid;gap:.25rem;font-size:.875rem;font-weight:680}
.check{display:flex;gap:.45rem;align-items:center;min-height:2.55rem}
select{appearance:none;font:inherit;color:var(--ink);padding:.46rem 2rem .46rem .7rem;border:1px solid #a4bcb6;border-radius:.4rem;background-color:#fff;max-width:100%;min-height:2.55rem;min-width:10rem;background-image:linear-gradient(45deg,transparent 50%,var(--green) 50%),linear-gradient(135deg,var(--green) 50%,transparent 50%);background-position:calc(100% - 1.05rem) 53%,calc(100% - .75rem) 53%;background-size:.31rem .31rem;background-repeat:no-repeat}
:focus-visible{outline:3px solid #167b6d;outline-offset:3px}
.panel{background:var(--white);border:1px solid var(--line);border-radius:.65rem;min-width:0;overflow:hidden;margin:1rem 0}
.plot{width:100%;min-height:375px}
.chart-note{padding:1rem;margin:0}
.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;text-align:left;font-size:.875rem;font-variant-numeric:tabular-nums;line-height:1.42}
td,th{padding:.65rem .75rem;white-space:nowrap;border-bottom:1px solid #e5edeb}
thead{background:#eaf3ef;color:#173c34}
tbody tr:nth-child(even){background:#f7faf9}
caption{text-align:left;font-weight:700;padding:.75rem}
details{padding:.65rem .9rem;border-top:1px solid var(--line)}
details:first-child{border-top:0}
summary{cursor:pointer;font-weight:700}
.doc{padding:.85rem 1rem}
.doc ul{margin:.35rem 0 0;padding-left:1.3rem}
.doc li+li{margin-top:.35rem}
footer{padding:1rem 0 2rem;border-top:1px solid var(--line)}
@media(max-width:450px){.toolbar label:not(.check),select{width:100%}.plot{min-height:340px}}
</style>
</head>
<body>
<header><div class="wrap">
<h1>__TITLE__</h1>
<p>__DESCRIPTION__</p>
<nav aria-label="Report navigation"><a href="../index.html">Latest results</a><a href="../setup/index.html">Algorithm setup</a><a href="../development/index.html">Development history</a><a href="../simulator/index.html">Simulator</a><a href="#training">Training profiles</a><a href="#held-out">Held-out results</a></nav>
</div></header>
<main>
<div class="toolbar layout-picker"><label>Instance<select id="instance"></select></label><p id="layout" aria-live="polite"></p><a id="routes" hidden>Inspect example routes</a></div>
<section id="training">
<h2>Training profiles</h2>
<p id="training-note"></p>
<div class="toolbar">
  <label>Metric<select id="profile-metric"></select></label>
  <label>Agent<select id="family"><option value="all">DQN and PPO</option><option value="dqn">DQN</option><option value="ppo">PPO</option></select></label>
  <label class="check"><input id="variation" type="checkbox" checked>Show seed variation</label>
</div>
<div class="panel"><div id="profiles" class="plot" aria-label="Training metric comparison"></div></div>
</section>
<section id="composition">
<h2>Reward composition</h2>
<p id="formula"></p>
<div class="toolbar"><label>Strategy<select id="strategy"></select></label></div>
<div class="panel"><div id="contributions" class="plot" aria-label="Weighted reward contributions during training"></div></div>
<div class="panel scroll"><table aria-label="Reward terms in the selected strategy's latest training window">
<caption id="window-caption"></caption>
<thead><tr><th scope="col">Term</th><th scope="col">Raw mean</th><th scope="col">Weight</th><th scope="col">Mean reward contribution</th><th scope="col">Sample SD in points</th></tr></thead>
<tbody id="term-rows"></tbody></table></div>
</section>
<section id="held-out">
<h2>Held-out results</h2>
<p id="held-out-note"></p>
<div class="toolbar"><label>Training seed<select id="training-seed"></select></label></div>
<div class="panel scroll"><table aria-label="Held-out episode outcomes">
<thead><tr><th scope="col">Strategy</th><th scope="col">Completed</th><th scope="col">Unserved</th><th scope="col">Travel min</th><th scope="col">Wait min</th><th scope="col" id="held-out-score">Mean reward</th><th scope="col">Reward SD</th><th scope="col" id="held-out-variant" hidden>Mean reward at training weights</th><th scope="col">Episodes</th></tr></thead>
<tbody id="held-out-rows"></tbody></table></div>
</section>
<section id="protocol">
<h2>Experiment setup</h2>
<div class="panel doc"><p id="protocol-note"></p><ul id="setup"></ul></div>
<div class="panel" id="methods"></div>
</section>
<footer><a href="../development/index.html">Development timeline</a></footer>
</main>
<script>__PLOTLY__</script>
<script type="application/json" id="exploration-data">__EXPERIMENTS__</script>
<script>
"use strict";
const $=id=>document.getElementById(id);
const experiments=JSON.parse($("exploration-data").textContent);
const defaultWeights={completed:10,unserved:-8,travel_minutes:-.03,waiting_minutes:-.05};
const terms=[
  {key:"completed",weighted:"completed_weighted",label:"Completed visits",unit:"visits",color:"#08776c"},
  {key:"unserved",weighted:"unserved_weighted",label:"Unserved visits",unit:"visits",color:"#a64f2b"},
  {key:"travel_minutes",weighted:"travel_weighted",label:"Travel",unit:"minutes",color:"#265caf"},
  {key:"waiting_minutes",weighted:"waiting_weighted",label:"Appointment waiting",unit:"minutes",color:"#855394"}
];
const metrics=[
  ["reward","Weighted reward sum","Reward points per episode"],
  ["reference_reward","Reward at original weights","Reward points per episode"],
  ...terms.map(t=>[t.key,t.label,t.unit[0].toUpperCase()+t.unit.slice(1)+" per episode"]),
  ...terms.map(t=>[t.weighted,t.label+" contribution","Reward points per episode"]),
  ["realized_reward","Realized episode reward","Reward points per episode"],
  ["train/loss","Training loss","Loss"],
  ["train/value_loss","Value loss","Loss"],
  ["train/policy_gradient_loss","Policy-gradient loss","Loss"],
  ["train/entropy_loss","Entropy loss","Loss"],
  ["train/approx_kl","Approximate KL divergence","KL divergence"]
];
const colors=["#08776c","#265caf","#99541e","#8e4691","#42642d","#b13e62","#445e70","#8d700a","#573a9b","#a74331","#317977","#6d5129"];
const numeric=n=>typeof n==="number" && Number.isFinite(n);
const fmt=(value,digits=2)=>numeric(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):"—";
const average=values=>values.length?values.reduce((sum,n)=>sum+n,0)/values.length:null;
const sd=values=>{
  if(values.length<2)return null;
  const mean=average(values);
  return Math.sqrt(values.reduce((sum,n)=>sum+(n-mean)**2,0)/(values.length-1));
};
const selected=()=>experiments[Number($("instance").value)];
const observations=(exp,id)=>exp.experiments[id]??{};
const weights=(exp,id)=>({...defaultWeights,...exp.meta.reward_weights,...exp.meta.variant_weights?.[id]});
const usesReferenceReward=exp=>Object.keys(exp.meta.variant_weights??{}).length>0;

function make(tag,text){
  const element=document.createElement(tag);
  if(text!==undefined)element.textContent=text;
  return element;
}
function options(id,entries,preferred){
  const node=$(id), previous=node.value;
  node.replaceChildren(...entries.map(([value,label])=>new Option(label,String(value))));
  const values=entries.map(([value])=>String(value));
  if(values.includes(previous))node.value=previous;
  else if(values.includes(String(preferred)))node.value=String(preferred);
}
function row(values){
  const tr=make("tr");
  for(const value of values)tr.append(make("td",value));
  return tr;
}
function series(exp,method,key){
  const profiles=observations(exp,method.id).profiles??{};
  if(profiles[key]?.length)return profiles[key];
  const term=terms.find(t=>t.weighted===key);
  if(!term)return [];
  const weight=weights(exp,method.id)[term.key];
  return (profiles[term.key]??[]).map(p=>({...p,mean:p.mean*weight,std:numeric(p.std)?p.std*Math.abs(weight):null}));
}
function transparent(color){
  const parts=[1,3,5].map(i=>parseInt(color.slice(i,i+2),16));
  return `rgba(${parts.join(",")},.13)`;
}
function traces(points,label,color,group,band=false,thick=false){
  const x=points.map(p=>p.steps), y=points.map(p=>p.mean);
  const result=[];
  const range=delta=>points.map(p=>numeric(p.std)&&p.n>1?p.mean+delta*p.std:null);
  if(band&&points.some(p=>numeric(p.std)&&p.n>1)){
    const base={x,type:"scatter",mode:"lines",line:{width:0},hoverinfo:"skip",showlegend:false,legendgroup:group};
    result.push({...base,y:range(-1)}, {...base,y:range(1),fill:"tonexty",fillcolor:transparent(color)});
  }
  result.push({x,y,type:"scatter",mode:"lines+markers",name:label,legendgroup:group,
    line:{color,width:thick?3:2},marker:{size:4},customdata:points.map(p=>[fmt(p.std,4),p.n]),
    hovertemplate:"%{x:,} steps<br>Mean %{y:.4g}<br>Sample SD %{customdata[0]}<br>Seeds %{customdata[1]}<extra>%{fullData.name}</extra>"});
  return result;
}
function chart(id,data,yaxis){
  const container=$(id);
  if(!data.length){
    if(container.data)Plotly.purge(container);
    container.replaceChildren(make("p","No profile was recorded for this selection."));
    container.classList.add("chart-note");
    return;
  }
  if(container.classList.contains("chart-note"))container.replaceChildren();
  container.classList.remove("chart-note");
  Plotly.react(container,data,{
    autosize:true,height:390,margin:{l:74,r:28,t:22,b:105},paper_bgcolor:"#fff",plot_bgcolor:"#fff",
    font:{family:"system-ui, sans-serif",color:"#152d35"},hovermode:"closest",
    xaxis:{title:{text:"Training steps"},gridcolor:"#eff4f2",tickformat:",d",automargin:true},
    yaxis:{title:{text:yaxis},gridcolor:"#e9f1ef",zerolinecolor:"#9db5ae",automargin:true},
    legend:{orientation:"h",y:-.28,groupclick:"togglegroup"}
  },{responsive:true,displaylogo:false});
}

function updateProfiles(){
  const exp=selected(), family=$("family").value, key=$("profile-metric").value;
  const data=exp.methods.flatMap((method,i)=>{
    if(family!=="all"&&method.family!==family)return [];
    const points=series(exp,method,key);
    return points.length?traces(points,method.label,colors[i%colors.length],method.id,$("variation").checked):[];
  });
  chart("profiles",data,metrics.find(metric=>metric[0]===key)?.[2]??"Value per episode");
}
function updateComposition(){
  const exp=selected(),method=exp.methods.find(m=>m.id===$("strategy").value);
  const reward=method?series(exp,method,"reward"):[], step=reward.at(-1)?.steps;
  const w=weights(exp,method?.id);
  $("formula").textContent=`${method?.label??"The selected strategy"} uses ${fmt(w.completed)} points per completed visit, ${fmt(w.unserved)} per unserved visit, ${fmt(w.travel_minutes)} per travel minute, and ${fmt(w.waiting_minutes)} per appointment waiting minute.`;
  const data=[];
  if(method){
    for(const term of terms){
      const points=series(exp,method,term.weighted);
      if(points.length)data.push(...traces(points,term.label,term.color,term.key));
    }
    if(reward.length)data.push(...traces(reward,"Weighted reward sum","#182b35","reward",false,true));
  }
  chart("contributions",data,"Reward points per episode");
  $("window-caption").textContent=step?`Training window ending at step ${fmt(step,0)}`:"Latest training window";
  const rows=method?terms.map(term=>{
    const raw=series(exp,method,term.key).find(p=>p.steps===step);
    const contribution=series(exp,method,term.weighted).find(p=>p.steps===step);
    return row([term.label,raw?`${fmt(raw.mean)} ${term.unit}`:"—",fmt(w[term.key]),fmt(contribution?.mean),fmt(contribution?.std)]);
  }):[];
  const sum=reward.at(-1);
  if(sum)rows.push(row(["Weighted reward sum","—","—",fmt(sum.mean),fmt(sum.std)]));
  $("term-rows").replaceChildren(...rows);
}

const outcomeKeys=["jobs_completed","jobs_unserved","travel_minutes","waiting_minutes","total_reward"];
function summarize(episodes,key){
  const groups=new Map();
  for(const episode of episodes){
    const value=episode.metrics?.[key];
    if(!numeric(value))continue;
    const seed=episode.training_seed??"fixed";
    if(!groups.has(seed))groups.set(seed,[]);
    groups.get(seed).push(value);
  }
  const means=[...groups.values()].map(average);
  return {mean:average(means),std:sd(means)};
}
function updateHeldOut(){
  const exp=selected(), seed=$("training-seed").value;
  const test=exp.meta.test_seeds??[];
  const common=usesReferenceReward(exp), reference=weights(exp);
  const sharedWaitCost=exp.meta.round==="guided-long-returns"
    ? Math.abs(Object.values(exp.meta.variant_weights??{})[0]?.waiting_minutes) : null;
  $("held-out-score").textContent=common?"Mean reference reward":"Mean reward";
  $("held-out-variant").hidden=!common;
  $("held-out-variant").textContent=numeric(sharedWaitCost)?`Mean reward at ${fmt(sharedWaitCost)} wait cost`:"Mean reward at training weights";
  $("held-out-note").textContent=`Test seeds ${test.join(", ")}. Each training seed receives equal weight. Reward SD describes the variation across training-seed means. Fixed policies use one evaluation per test seed.`;
  if(common){
    $("held-out-note").textContent+=` Reference reward evaluates every schedule with the original weights, including ${fmt(reference.waiting_minutes)} points per waiting minute.`;
    $("held-out-note").textContent+=numeric(sharedWaitCost)
      ? ` The second score applies a cost of ${fmt(sharedWaitCost)} points per waiting minute to every schedule. Both scores for Random and ACO use their saved held-out schedules.`
      : " The training-weight column uses each strategy's own weights.";
  }
  $("held-out-rows").replaceChildren(...exp.methods.map(method=>{
    const recorded=observations(exp,method.id).episodes??[];
    const runs=recorded.filter(ep=>method.family==="baseline"||!seed||String(ep.training_seed)===seed);
    const values=outcomeKeys.map(key=>summarize(runs,key));
    const tr=row([method.label,...values.map(value=>fmt(value.mean)),fmt(values.at(-1).std)]);
    if(common)tr.append(make("td",fmt(summarize(runs,"raw_variant_reward").mean)));
    tr.append(make("td",fmt(runs.length,0)));
    return tr;
  }));
}
function updateSetup(){
  const exp=selected(), meta=exp.meta;
  $("protocol-note").textContent=meta.note??"";
  const facts=[
    meta.steps&&`Up to ${fmt(meta.steps,0)} learning steps per training seed.`,
    meta.training_seeds?.length&&`Training seeds ${meta.training_seeds.join(", ")}.`,
    meta.demo_seeds?.length&&`PPO demonstration seeds ${meta.demo_seeds.join(", ")}.`,
    meta.validation_seeds?.length&&`Validation seeds ${meta.validation_seeds.join(", ")}.`,
    meta.devices&&Object.entries(meta.devices).filter(([,value])=>typeof value==="string").map(([key,value])=>`${key.toUpperCase()} used ${value.toUpperCase()}.`).join(" "),
    meta.family_times&&Object.entries(meta.family_times).filter(([,value])=>numeric(value)).map(([key,value])=>`${key.toUpperCase()} experiment took ${fmt(value,1)} seconds.`).join(" ")
  ].filter(Boolean);
  $("setup").replaceChildren(...facts.map(fact=>make("li",fact)));
  $("methods").replaceChildren(...exp.methods.map(method=>{
    const details=make("details"), description=make("p",method.description);
    details.append(make("summary",method.label),description);
    const setting=meta.variant_settings?.[method.id];
    if(setting){
      const names={width:"Network width",batch_size:"Batch size",learning_rate:"Learning rate",target_update_interval:"Target update interval",exploration_fraction:"Exploration fraction",ent_coef:"Entropy coefficient",gamma:"Discount γ",gae_lambda:"GAE λ",n_steps:"Rollout length"};
      const params=Object.entries(setting.params??{}).filter(([key])=>names[key]);
      if(params.length){
        const values=params.map(([key,value])=>`${names[key]} ${value}`);
        if(method.family==="dqn"&&numeric(setting.return_steps))values.push(`Return length ${setting.return_steps}`);
        details.append(make("p",values.join(" · ")));
      }
    }
    const reps=observations(exp,method.id).replicates??[];
    if(!reps.length)return details;
    const frame=make("div"), table=make("table");
    frame.className="scroll";
    table.setAttribute("aria-label",`${method.label} training runs`);
    const guided=reps.some(r=>Object.keys(r.guide_counts??{}).length);
    const imitated=reps.some(r=>r.demo_stats);
    const labels={short_wait:"short-wait",aco:"ACO",uniform:"uniform",forced_defer:"forced defer"};
    const heading=table.createTHead().insertRow();
    const titles=["Seed","Steps","Completed training episodes","Seconds"];
    if(guided)titles.push("Exploratory actions by source");
    if(imitated)titles.push("Demonstration examples","Imitation loss, before → after");
    for(const title of titles){
      const th=make("th",title);
      th.scope="col";
      heading.append(th);
    }
    const body=table.createTBody();
    for(const r of reps){
      const cells=[r.seed,fmt(r.steps,0),fmt(r.episodes,0),fmt(r.seconds)];
      if(guided)cells.push(Object.entries(r.guide_counts??{}).map(([source,n])=>`${fmt(n,0)} ${labels[source]??source}`).join(", "));
      if(imitated)cells.push(fmt(r.demo_examples,0),`${fmt(r.demo_stats?.loss_before)} → ${fmt(r.demo_stats?.loss_after)}`);
      body.append(row(cells));
    }
    frame.append(table);
    details.append(frame);
    return details;
  }));
}
function updateInstance(){
  const exp=selected(), methods=exp.methods.filter(method=>series(exp,method,"reward").length);
  $("layout").textContent=`${fmt(exp.meta.jobs,0)} appointments and ${fmt(exp.meta.nurses,0)} nurses`;
  $("routes").hidden=!exp.meta.routes_href;
  if(exp.meta.routes_href)$("routes").href=exp.meta.routes_href;
  $("training-note").textContent=(exp.meta.rollout_window?`Reward profiles use the latest ${exp.meta.rollout_window} completed training episodes per seed at each recorded step. `:"")+"Bands show one sample standard deviation on either side of the mean across training seeds. Hover for the sample size.";
  if(usesReferenceReward(exp))$("training-note").textContent+=exp.meta.round==="guided-long-returns"
    ? " Weighted training metrics use the 0.15 waiting cost. The reference reward profile uses the original 0.05 cost."
    : " Weighted metrics use each strategy's training weights. Compare completed visits and raw waiting minutes directly across weight settings.";
  options("profile-metric",metrics.filter(([key])=>exp.methods.some(method=>series(exp,method,key).length)).map(([key,label])=>[key,label]),usesReferenceReward(exp)?"reference_reward":"reward");
  options("strategy",methods.map(method=>[method.id,method.label]));
  options("training-seed",[["","All training seeds"],...(exp.meta.training_seeds??[]).map(seed=>[seed,`Seed ${seed}`])]);
  updateProfiles();
  updateComposition();
  updateHeldOut();
  updateSetup();
}
options("instance",experiments.map((exp,i)=>[i,`Instance ${exp.meta.instance}`]));
$("instance").addEventListener("change",updateInstance);
for(const id of ["profile-metric","family","variation"])$(id).addEventListener("change",updateProfiles);
$("strategy").addEventListener("change",updateComposition);
$("training-seed").addEventListener("change",updateHeldOut);
updateInstance();
</script>
</body>
</html>
"""
