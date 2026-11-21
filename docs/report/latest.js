"use strict";

const data = document.querySelector("#latest-data");
const panels = JSON.parse(data.textContent);
const selectLayout = document.querySelector("#layout");
const selectMetric = document.querySelector("#metric");
const metricNames = {
  raw_variant_reward: `Reward at ${data.dataset.waitingCost ?? "0.15"}`,
  total_reward: "Reward at 0.05",
  jobs_completed: "Visits completed",
  jobs_unserved: "Visits unserved",
  waiting_minutes: "Waiting minutes",
  travel_minutes: "Travel minutes"
};
const format = new Intl.NumberFormat("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

for (const panel of panels) selectLayout.add(new Option(`Instance ${panel.instance}`, panel.instance));
for (const [key, label] of Object.entries(metricNames)) selectMetric.add(new Option(label, key));
if (panels.some(panel => panel.instance === "D")) selectLayout.value = "D";

function element(tag, value) {
  const node = document.createElement(tag);
  node.textContent = value;
  return node;
}

function render() {
  const panel = panels.find(item => item.instance === selectLayout.value);
  const key = selectMetric.value;
  document.querySelector("#layout-note").textContent = `${panel.jobs} appointments · ${panel.nurses} nurses`;
  document.querySelector("#routes-link").href = data.dataset.waitingCost ? panel.routes : `guided-returns/${panel.routes}`;
  const lead = Math.max(...panel.rows.filter(row => row.family !== "baseline").map(row => row.raw_variant_reward));
  const values = panel.rows.map(row => row[key]);
  const min = Math.min(0, ...values);
  const width = Math.max(0, ...values) - min || 1;
  const percentage = value => 100 * (value - min) / width;
  const zero = percentage(0);
  const fields = ["jobs_completed", "jobs_unserved", "travel_minutes", "waiting_minutes", "raw_variant_reward", "total_reward"];

  document.querySelector("#policy-results").replaceChildren(...panel.rows.map(row => {
    const tr = document.createElement("tr");
    tr.dataset.leading = row.family !== "baseline" && row.raw_variant_reward === lead;
    const label = element("th", row.label);
    label.scope = "row";
    tr.append(label, ...fields.map(field => element("td", format.format(row[field]))));
    return tr;
  }));

  document.querySelector("#policy-chart").replaceChildren(...panel.rows.map(row => {
    const group = document.createElement("div");
    group.className = "result-bar";
    group.dataset.group = row.family;
    const track = document.createElement("span");
    const fill = document.createElement("span");
    const origin = document.createElement("span");
    track.className = "track";
    fill.className = "fill";
    origin.className = "zero";
    const at = percentage(row[key]);
    fill.style.left = `${Math.min(zero, at)}%`;
    fill.style.width = `${Math.abs(zero - at)}%`;
    origin.style.left = `${zero}%`;
    track.append(fill, origin);
    group.append(element("span", row.label), track, element("strong", format.format(row[key])));
    return group;
  }));
}

selectLayout.addEventListener("change", render);
selectMetric.addEventListener("change", render);
render();
