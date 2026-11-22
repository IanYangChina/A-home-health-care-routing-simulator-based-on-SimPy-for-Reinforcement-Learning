"""Build an offline overview linking the full reports for each benchmark run."""
from __future__ import annotations

import json
from pathlib import Path


def render_suite(payloads: list[dict], outputs: list[str], output: Path, *, project_prefix: str = '') -> Path:
    if not payloads or len(payloads) != len(outputs):
        raise ValueError("Provide one report link for each benchmark payload.")

    experiments = [
        {
            "meta": payload["meta"],
            "methods": payload["methods"],
            "networks": {
                method: settings.get("best_params", {}).get("width")
                for method, settings in payload.get("tuning", {}).items()
            },
            "runs": [
                {key: episode[key] for key in ("method", "seed", "metrics")}
                for episode in payload["runs"]
            ],
            "report": report,
        }
        for payload, report in zip(payloads, outputs)
    ]
    data = json.dumps(experiments, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = _HTML.replace("__PROJECT__", project_prefix).replace("__EXPERIMENTS__", data.replace("<", r"\u003c"))
    output.write_text(html, encoding="utf-8")
    return output


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Compare home care dispatch policies across appointment layouts and inspect reinforcement-learning training results.">
<title>Home care dispatch benchmark</title>
<style>
:root{color-scheme:light;--ink:#152d35;--muted:#4d6269;--line:#d7e2df;--page:#f3f7f5;--fixed:#08776c;--rl:#2852a0;--selected:#985220}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:1rem/1.55 system-ui,-apple-system,sans-serif}
main,.wrap{max-width:1300px;margin:auto;padding-inline:clamp(1rem,4vw,3rem)}
header{background:#123831;color:#fff;border-bottom:4px solid #58b8a0}
header .wrap{padding-block:clamp(2rem,6vw,4.5rem)}
header p{max-width:720px;color:#dbefea;margin:1rem 0 1.5rem}
h1,h2{margin:0;font-weight:720;line-height:1.12;letter-spacing:-.035em}
h1{font-size:clamp(2.2rem,6vw,4.1rem)}
h2{font-size:clamp(1.65rem,3vw,2.3rem)}
a{color:#165d51;text-underline-offset:.2em}
header a{color:#ddf7ef}
nav,.toolbar{display:flex;flex-wrap:wrap;gap:.75rem 1.3rem}
section{margin-block:clamp(2rem,4.5vw,3.6rem);scroll-margin-top:1rem}
section>p{max-width:900px;color:var(--muted);margin:.6rem 0 1rem}
.toolbar{margin:1rem 0}
label{display:grid;gap:.25rem;font-size:.875rem;font-weight:680;min-width:10rem}
select{appearance:none;font:inherit;color:var(--ink);padding:.46rem 2rem .46rem .7rem;border:1px solid #a4bcb6;border-radius:.4rem;background-color:#fff;max-width:100%;min-height:2.55rem;background-image:linear-gradient(45deg,transparent 50%,#165d51 50%),linear-gradient(135deg,#165d51 50%,transparent 50%);background-position:calc(100% - 1.05rem) 53%,calc(100% - .75rem) 53%;background-size:.31rem .31rem;background-repeat:no-repeat}
:focus-visible{outline:3px solid #167b6d;outline-offset:3px}
.panel{min-width:0;background:#fff;border:1px solid var(--line);border-radius:.65rem;margin:1rem 0}
.scroll{overflow:auto}
.chart{padding:1rem}
.bar-row{display:grid;grid-template-columns:clamp(7rem,20vw,12rem) minmax(2rem,1fr) 5.25rem;align-items:center;gap:.6rem;margin:.38rem 0}
.bar-row span,.key{font-size:.875rem}
.bar-row strong{text-align:right;font-variant-numeric:tabular-nums}
.track{height:.85rem;background:#edf3f1;border-radius:.25rem;position:relative}
.bar,.origin{position:absolute;height:100%;top:0}
.origin{border-left:1px solid #778c85}
.bar,.key i{background:var(--fixed);border-radius:.2rem}
.bar[data-kind=rl],.key i[data-kind=rl]{background:var(--rl)}
.bar[data-kind=selected],.key i[data-kind=selected]{background:var(--selected)}
.key{display:flex;gap:.5rem 1rem;flex-wrap:wrap;color:var(--muted);margin:.85rem 0 0}
.key i{display:inline-block;width:.75rem;height:.75rem;margin-right:.28rem}
table{border-collapse:collapse;width:100%;text-align:left;font-size:.875rem;font-variant-numeric:tabular-nums;line-height:1.45}
td,th{padding:.65rem .75rem;white-space:nowrap;border-bottom:1px solid #e5edeb}
thead{background:#eaf3ef;color:#173c34}
tbody tr:nth-child(even){background:#f7faf9}
tr[data-selected=true]{background:#f8eee5}
.summary{padding:.75rem 1rem;background:#e8f2ef;border-radius:.45rem;margin:1rem 0}
footer{border-top:1px solid var(--line);padding:1rem 0 2rem;color:var(--muted)}
@media(max-width:510px){.toolbar label{width:100%}.chart{padding:.8rem}.bar-row{grid-template-columns:6rem minmax(1rem,1fr) 4rem;gap:.35rem}}
</style>
</head>
<body>
<header><div class="wrap">
<h1>Home care dispatch benchmark</h1>
<p>Compare how fixed heuristics and reinforcement-learning agents assign nurses to home visits. Explore outcomes across appointment layouts or open a run to see its routes, schedules, and job assignments.</p>
<nav aria-label="Report sections"><a href="__PROJECT__index.html">Home</a><a href="__PROJECT__results.html">Latest results</a><a href="__PROJECT__setup/index.html">Algorithm setup</a><a href="__PROJECT__development/index.html">Development history</a><a href="__PROJECT__simulator/index.html">Simulator</a><a href="#policies">Policy results</a><a href="#width" id="width-link" hidden>Network width</a><a href="#reports">Detailed reports</a></nav>
</div></header>
<main>
<section id="policies">
<h2>Policy results</h2>
<div class="toolbar">
  <label>Experiment <select id="experiment"></select></label>
  <label>Metric <select id="metric"></select></label>
  <label>Test seed <select id="seed"></select></label>
</div>
<p id="design"></p>
<div class="summary" id="selection" aria-live="polite"></div>
<div class="panel chart">
  <div id="bars" aria-label="Policy metric comparison"></div>
  <div class="key"><span><i></i>Fixed policy</span><span><i data-kind="rl"></i>Trained agent</span><span><i data-kind="selected"></i>Validation-selected agent</span></div>
</div>
<div class="panel scroll"><table aria-label="Policy results">
  <thead><tr><th scope="col">Policy</th><th scope="col">Network</th><th scope="col">Visits</th><th scope="col">Completion</th><th scope="col">Reward</th><th scope="col">Travel min</th><th scope="col">Wait min</th><th scope="col">Metric SD</th><th scope="col">Episodes</th></tr></thead>
  <tbody id="results"></tbody>
</table></div>
<p>Metric SD is the sample standard deviation of the selected metric across the displayed test episodes.</p>
<a id="detail" href="#reports">Inspect individual routes and schedules</a>
</section>
<section id="width" hidden>
<h2>Network width on instance A</h2>
<p id="width-design"></p>
<div class="toolbar">
  <label>Metric <select id="width-metric"></select></label>
  <label>Paired test seed <select id="width-seed"></select></label>
</div>
<div class="panel scroll"><table aria-label="Paired network width results">
  <thead><tr><th scope="col">Policy</th><th scope="col" id="narrow"></th><th scope="col" id="wide"></th><th scope="col">Change (wider − narrower)</th><th scope="col">Paired change SD</th><th scope="col">Paired seeds</th></tr></thead>
  <tbody id="width-results"></tbody>
</table></div>
<p>Each row compares the same policy on shared test seeds. The reported spread is across episodes for one training seed.</p>
<p id="runtime-note" hidden>The original A run predates a simulator speed improvement. Its evaluation runtime also reflects that implementation difference.</p>
</section>
<section id="reports">
<h2>Detailed reports</h2>
<p>Each report contains recorded routes, nurse schedules, job allocations, event traces, and training settings. Training diagnostics include loss and reward profiles across available training seeds.</p>
<div class="panel scroll"><table aria-label="Experiment reports and training budgets">
  <thead><tr><th scope="col">Report</th><th scope="col">Appointments</th><th scope="col">Nurses</th><th scope="col">Optuna per agent</th><th scope="col">Final steps per training seed</th><th scope="col">Training seeds</th><th scope="col">Test seeds</th><th scope="col">Best fixed reward</th><th scope="col">Selected RL reward</th></tr></thead>
  <tbody id="report-links"></tbody>
</table></div>
<p>Rewards are means across test seeds. RL agents were trained separately on each layout, and final RL selection uses validation reward.</p>
</section>
<footer>Home care routing and scheduling research simulation</footer>
</main>
<noscript>Enable JavaScript to see the embedded comparisons.</noscript>
<script type="application/json" id="experiments">__EXPERIMENTS__</script>
<script>
const $ = id => document.getElementById(id);
const experiments = JSON.parse($("experiments").textContent);
const metrics = {
  jobs_completed: ["Visits completed", 2], completion_rate: ["Completion (%)", 1, 100],
  total_reward: ["Reward", 2], travel_minutes: ["Travel (nurse-minutes)", 1],
  waiting_minutes: ["Waiting (nurse-minutes)", 1], service_minutes: ["Service (nurse-minutes)", 1],
  time: ["Shift elapsed (minutes)", 1], decisions: ["Dispatch decisions", 1],
  wall_seconds: ["Evaluation runtime (seconds)", 3], truncated: ["Shift truncation (%)", 1, 100]
};
const presets = Object.entries(metrics).map(([key, [label]]) => [key, label]);
const shown = ["jobs_completed", "completion_rate", "total_reward", "travel_minutes", "waiting_minutes"];
const value = (run, key) => Number(run.metrics[key]) * (metrics[key][2] ?? 1);
const average = values => values.length ? values.reduce((sum, n) => sum + n, 0) / values.length : null;
function sd(values) {
  if (values.length < 2) return null;
  const mean = average(values);
  return Math.sqrt(values.reduce((sum, n) => sum + (n - mean) ** 2, 0) / (values.length - 1));
}
const format = (number, digits = 0) => number == null || !Number.isFinite(number) ? "—"
  : number.toLocaleString("en", {minimumFractionDigits:digits,maximumFractionDigits:digits});
const width = exp => (exp.meta.network ?? [64, 64]).join(" × ");
const label = exp => "Instance " + exp.meta.instance + " · " + (exp.meta.network_search?.length ? "Tuned networks" : width(exp));
const network = (exp, method) => {
  if (method.group !== "rl") return "—";
  const units = exp.networks[method.id];
  return units == null ? width(exp) : Array.isArray(units) ? units.join(" × ") : `${units} × ${units}`;
};
const trainingCount = exp => exp.meta.training_seeds?.length ?? 1;
const budget = exp => exp.meta.trials
  ? `${exp.meta.trials} trials × ${format(exp.meta.trial_steps)} steps`
  : exp.meta.parameters_from ? `Parameters from ${exp.meta.parameters_from}` : "Fixed parameters";
const choose = (id, entries, selected) => {
  $(id).replaceChildren(...entries.map(([value, text]) => new Option(text, value)));
  if (selected != null && entries.some(([value]) => String(value) === String(selected))) $(id).value = selected;
};
const seedOptions = seeds => [["", "All test seeds"], ...seeds.map(seed => [seed, seed])];
const element = (tag, text = "") => Object.assign(document.createElement(tag), {textContent:text});
function row(values, selected = false) {
  const tr = element("tr");
  tr.dataset.selected = selected;
  values.forEach((item, index) => {
    const cell = tr.appendChild(element(index ? "td" : "th"));
    if (!index) cell.scope = "row";
    cell.append(item);
  });
  return tr;
}
const subset = (exp, method, seed) => exp.runs.filter(run => run.method === method && (!seed || String(run.seed) === seed));
const display = (number, key) => format(number, metrics[key][1]) + (number != null && metrics[key][2] ? "%" : "");
const rewardWinner = methods => methods.filter(item => item.method.group === "baseline").reduce((best, item) =>
  best == null || item.reward > best.reward ? item : best, null);

function renderPolicies() {
  const exp = experiments[Number($("experiment").value)];
  const key = $("metric").value;
  const seed = $("seed").value;
  const methods = exp.methods.map(method => {
    const runs = subset(exp, method.id, seed);
    return {method, runs, values:runs.map(run => value(run,key)), reward:average(runs.map(run => value(run,"total_reward")))};
  });
  const fixed = rewardWinner(methods);
  const selected = methods.find(item => item.method.id === exp.meta.best_rl);
  $("design").textContent = `${exp.meta.jobs} appointments, ${exp.meta.nurses} nurses, ${exp.meta.horizon}-minute shift. ${budget(exp)}. ${format(exp.meta.final_steps)} final steps per training seed and ${trainingCount(exp)} training seed(s) per agent.`;
  $("selection").textContent = selected && fixed
    ? `${selected.method.label} was selected on validation. ${seed ? "Test seed " + seed : "Mean test"} reward ${format(selected.reward,2)}. ${fixed.method.label} led the fixed policies with ${format(fixed.reward,2)}.`
    : "Policy results for the selected experiment.";
  $("detail").href = exp.report;
  const means = methods.map(item => average(item.values));
  const min = Math.min(0, ...means);
  const max = Math.max(0, ...means, key === "completion_rate" ? 100 : 0);
  const position = number => 100 * (number - min) / (max - min || 1);
  const zero = position(0);
  $("bars").replaceChildren(...methods.map(({method}, index) => {
    const mean = means[index];
    const group = element("div");
    group.className = "bar-row";
    const track = element("div");
    track.className = "track";
    track.setAttribute("aria-hidden", "true");
    const origin = element("span");
    origin.className = "origin";
    origin.style.left = `${Math.min(zero, 99.8)}%`;
    const bar = element("span");
    bar.className = "bar";
    bar.dataset.kind = method.id === exp.meta.best_rl ? "selected" : method.group;
    bar.style.left = `${Math.min(position(mean), zero)}%`;
    bar.style.width = `${Math.abs(position(mean) - zero)}%`;
    track.append(origin, bar);
    group.append(element("span", method.label), track, element("strong", display(mean,key)));
    return group;
  }));
  $("results").replaceChildren(...methods.map(({method,runs,values}) => row([
    method.label, network(exp,method),
    ...shown.map(metric => display(average(runs.map(run => value(run,metric))),metric)),
    display(sd(values),key), String(runs.length)
  ], method.id === exp.meta.best_rl)));
}

function renderWidths(narrow, wide) {
  const key = $("width-metric").value;
  const seed = $("width-seed").value;
  $("width-results").replaceChildren(...narrow.methods.map(method => {
    const wideRuns = new Map(subset(wide,method.id,seed).map(run => [run.seed,value(run,key)]));
    const pairs = subset(narrow,method.id,seed).filter(run => wideRuns.has(run.seed))
      .map(run => [value(run,key),wideRuns.get(run.seed)]);
    const changes = pairs.map(([oldValue,newValue]) => newValue - oldValue);
    const change = average(changes);
    const percentPoint = metrics[key][2] && change != null ? " pp" : "";
    return row([
      method.label,
      display(average(pairs.map(([oldValue]) => oldValue)),key),
      display(average(pairs.map(([,newValue]) => newValue)),key),
      (change > 0.5 * 10 ** -metrics[key][1] ? "+" : "") + format(change,metrics[key][1]) + percentPoint,
      format(sd(changes),metrics[key][1]) + (changes.length > 1 ? percentPoint : ""), String(pairs.length)
    ]);
  }));
}

const latestA = experiments.findLastIndex(exp => exp.meta.instance === "A");
choose("experiment", experiments.map((exp, index) => [index,label(exp)]), Math.max(latestA,0));
choose("metric", presets, "jobs_completed");
function changeExperiment() {
  const exp = experiments[Number($("experiment").value)];
  choose("seed", seedOptions(exp.meta.test_seeds), $("seed").value);
  renderPolicies();
}
$("experiment").onchange = changeExperiment;
$("metric").onchange = $("seed").onchange = renderPolicies;
changeExperiment();

const widths = experiments.filter(exp => exp.meta.instance === "A")
  .sort((one,two) => (one.meta.network ?? [64])[0] - (two.meta.network ?? [64])[0]);
if (widths.length === 2 && width(widths[0]) !== width(widths[1])) {
  const [narrow,wide] = widths;
  const shared = narrow.meta.test_seeds.filter(seed => wide.meta.test_seeds.includes(seed));
  $("width").hidden = $("width-link").hidden = false;
  $("narrow").textContent = width(narrow);
  $("wide").textContent = width(wide);
  const final = narrow.meta.final_steps === wide.meta.final_steps
    ? `Both widths used ${format(wide.meta.final_steps)} final training steps per agent.`
    : `Final training steps per agent were ${format(narrow.meta.final_steps)} and ${format(wide.meta.final_steps)}.`;
  $("width-design").textContent = (wide.meta.parameters_from
    ? `The ${width(wide)} networks reused the tuned ${width(narrow)} hyperparameters. ` : "") + final;
  $("runtime-note").hidden = narrow.meta.network != null;
  choose("width-metric", presets, "total_reward");
  choose("width-seed", seedOptions(shared));
  $("width-metric").onchange = $("width-seed").onchange = () => renderWidths(narrow,wide);
  renderWidths(narrow,wide);
}

$("report-links").replaceChildren(...experiments.map(exp => {
  const link = element("a", label(exp));
  link.href = exp.report;
  const results = exp.methods.map(method => ({method,
    reward:average(subset(exp, method.id, "").map(run => value(run,"total_reward")))}));
  const rewardLabel = item => item ? `${item.method.label} · ${format(item.reward,2)}` : "—";
  return row([link, String(exp.meta.jobs), String(exp.meta.nurses), budget(exp),
    format(exp.meta.final_steps), String(trainingCount(exp)), String(exp.meta.test_seeds.length),
    rewardLabel(rewardWinner(results)), rewardLabel(results.find(item => item.method.id === exp.meta.best_rl))]);
}));
</script>
</body>
</html>
"""
