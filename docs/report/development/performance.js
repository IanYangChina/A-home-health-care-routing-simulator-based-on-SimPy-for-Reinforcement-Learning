"use strict";

const {methods, metrics, rounds, instances} = window.developmentPerformance;
const charts = [...document.querySelectorAll(".performance-chart")];
const metricMenu = document.getElementById("performance-metrics");
const roundIds = rounds.map(round => round.number);
const timed = rounds.filter(round => round.start && round.end);
const metricNames = {
  jobs_completed: "Served visits", jobs_unserved: "Unserved visits", travel_minutes: "Travel",
  waiting_minutes: "Waiting", completion_percent: "Completion %", service_minutes: "Service",
  time: "Shift elapsed", decisions: "Decisions", reward_030: "Reward (0.30)", total_reward: "Reward (0.05)"
};
const palette = {
  random: ["#586977", "circle"], greedy: ["#8b5629", "diamond"], edf: ["#783b8e", "triangle-up"],
  aco: ["#08776c", "square"], dqn: ["#2657a5", "cross"], ppo: ["#b73567", "triangle-down"]
};
const dateFormat = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC", day: "2-digit", month: "short", year: "numeric",
  hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23"
});
const numberFormat = new Intl.NumberFormat("en-GB", {maximumFractionDigits: 3});
const rewardFormat = new Intl.NumberFormat("en-GB", {minimumFractionDigits: 2, maximumFractionDigits: 2});
const utc = iso => iso ? dateFormat.format(new Date(iso)) : "—";
const number = value => value == null ? "—" : numberFormat.format(value);
const elapsed = round => round.start && round.end ? (Date.parse(round.end) - Date.parse(round.start)) / 1000 : null;

function duration(seconds) {
  if (seconds == null) return "—";
  const hours = Math.floor(seconds / 3600);
  return [hours ? `${hours} h` : "", `${Math.floor(seconds / 60) % 60} min`, `${seconds % 60} s`].filter(Boolean).join(" ");
}

function tableRow(label, cells) {
  const row = document.createElement("tr");
  const heading = document.createElement("th");
  heading.scope = "row";
  heading.append(label);
  row.append(heading);
  for (const value of cells) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
  return row;
}

const estimated = timed.filter(round => round.estimated).length;
document.getElementById("performance-total").textContent = `${estimated ? "≈ " : ""}${duration(timed.reduce((total, round) => total + elapsed(round), 0))}`;
document.getElementById("performance-time-coverage").textContent = `${timed.length} of ${rounds.length} rounds timed${estimated ? ` · ${estimated} estimated intervals` : ""}.`;
const computeRounds = rounds.filter(round => round.compute_seconds != null);
const computeHours = computeRounds.reduce((total, round) => total + round.compute_seconds, 0) / 3600;
document.getElementById("performance-compute-total").textContent = computeRounds.length ? `${computeHours.toFixed(2)} h` : "—";
document.getElementById("performance-compute-coverage").textContent = `${computeRounds.length} of ${rounds.length} rounds have runner durations. Parallel runs counted separately.`;

function latestPerformance(layout) {
  const points = Object.entries(instances[layout]).flatMap(([method, samples]) => samples.map(point => ({method, ...point})));
  const latest = Math.max(...points.map(point => point.round));
  const ranked = points.filter(point => point.round === latest).sort((a, b) => b.metrics.reward_030.mean - a.metrics.reward_030.mean);
  const [best] = ranked;
  const learned = ranked.find(point => point.method === "dqn" || point.method === "ppo");
  const card = document.createElement("div");
  const heading = document.createElement("dt");
  heading.textContent = `Instance ${layout} · Round ${latest}`;
  card.append(heading);

  const score = document.createElement("dd");
  const mean = document.createElement("strong");
  mean.textContent = rewardFormat.format(best.metrics.reward_030.mean);
  const sd = best.metrics.reward_030.sd;
  score.append(mean, sd == null ? "" : ` ± ${rewardFormat.format(sd)}`);
  card.append(score);

  const variant = document.createElement("dd");
  variant.className = "caption";
  variant.textContent = best.variant;
  card.append(variant);
  if (learned && learned !== best) {
    const bestLearned = document.createElement("dd");
    bestLearned.className = "caption";
    bestLearned.textContent = `Best learned ${methods[learned.method]} · ${rewardFormat.format(learned.metrics.reward_030.mean)}`;
    card.append(bestLearned);
  }
  return card;
}

document.getElementById("performance-best").replaceChildren(...Object.keys(instances).map(latestPerformance));

document.getElementById("performance-times").replaceChildren(...rounds.map(round => {
  const link = document.createElement("a");
  link.href = `#round-${round.number}`;
  link.textContent = `Round ${round.number}`;
  link.title = round.note;
  return tableRow(link, [utc(round.start), utc(round.end), `${round.estimated ? "≈ " : ""}${duration(elapsed(round))}`, round.layouts.join(", ") || "—"]);
}));

function tracesFor(method, points, key) {
  const [color, symbol] = palette[method];
  const lookup = new Map(points.map(point => [point.round, point]));
  const samples = rounds.map(round => lookup.get(round.number)?.metrics[key]);
  const deviations = samples.map(sample => sample?.sd ?? null);
  const means = samples.map(sample => sample?.mean ?? null);
  const rgb = color.slice(1).match(/../g).map(hex => parseInt(hex, 16));
  const bandX = [], bandY = [];
  let segment = [];
  for (let i = 0; i <= samples.length; i++) {
    if (samples[i]?.sd != null) {
      segment.push(i);
      continue;
    }
    if (segment.length > 1) {
      const reverse = [...segment].reverse();
      bandX.push(...segment.map(j => roundIds[j]), ...reverse.map(j => roundIds[j]), null);
      bandY.push(...segment.map(j => means[j] - deviations[j]), ...reverse.map(j => means[j] + deviations[j]), null);
    }
    segment = [];
  }
  const shared = {type: "scatter", connectgaps: false, legendgroup: method};
  return [
    {...shared, x: bandX, y: bandY, mode: "lines", line: {width: 0}, showlegend: false, hoverinfo: "skip",
      fill: "toself", fillcolor: `rgba(${rgb.join(",")},0.10)`},
    {...shared, x: roundIds, y: means, name: methods[method], mode: "lines+markers", line: {color, width: 2},
      marker: {symbol, size: 9, color},
      error_y: {type: "data", array: deviations, visible: true, color, thickness: 1, width: 3},
      customdata: rounds.map(round => {
        const point = lookup.get(round.number);
        return point ? [round.number, point.variant, number(point.metrics[key].mean), number(point.metrics[key].sd),
          point.n, utc(round.start), utc(round.end), `${round.estimated ? "≈ " : ""}${duration(elapsed(round))}`] : null;
      }),
      hovertemplate: "Round %{customdata[0]} · %{customdata[1]}<br>Mean %{customdata[2]}<br>Sample SD %{customdata[3]}<br>Test shifts %{customdata[4]}<br>Start %{customdata[5]} UTC<br>End %{customdata[6]} UTC<br>Elapsed %{customdata[7]}<extra></extra>"}
  ];
}

function renderPerformance(key) {
  for (const button of metricMenu.children) button.setAttribute("aria-pressed", String(button.dataset.metric === key));
  for (const chart of charts) {
    const layout = chart.dataset.instance;
    const available = instances[layout];
    const data = Object.keys(methods).filter(method => available[method]?.length)
      .flatMap(method => tracesFor(method, available[method], key));
    Plotly.react(chart, data, {
      height: 310, margin: {l: 60, r: 14, t: 45, b: 46}, uirevision: layout,
      font: {family: "system-ui, sans-serif", color: "#152d35", size: 12},
      xaxis: {type: "linear", title: {text: "Round ID"}, tickmode: "array", tickvals: roundIds,
        range: [roundIds[0] - .45, roundIds.at(-1) + .45], fixedrange: true, gridcolor: "#eef3f1"},
      yaxis: {title: {text: metrics[key]}, automargin: true, uirevision: key,
        gridcolor: "#e4eeeb", zerolinecolor: "#a5b8b6"},
      legend: {orientation: "h", y: 1.03, yanchor: "bottom", font: {size: 10}, groupclick: "togglegroup", traceorder: "normal"},
      hovermode: "closest", dragmode: false
    }, {responsive: true, displayModeBar: false, scrollZoom: false});
  }
}

for (const [key, label] of Object.entries(metrics)) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.metric = key;
  button.textContent = metricNames[key] ?? label;
  button.title = label;
  button.addEventListener("click", () => renderPerformance(key));
  metricMenu.append(button);
}
renderPerformance("jobs_completed");
