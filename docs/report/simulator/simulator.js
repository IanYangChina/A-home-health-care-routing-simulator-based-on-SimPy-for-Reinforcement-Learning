"use strict";

(() => {
  const element = id => document.getElementById(id);
  const {scenarios, baseService} = globalThis.DISPATCH_DATA;
  const minute = value => value == null ? "—" : value.toFixed(1);
  const visit = id => id == null ? "—" : `#${id}`;
  const nurseName = id => id == null ? "—" : `Nurse ${id + 1}`;
  let simulator;

  function start() {
    simulator = new globalThis.DispatchSimulator(scenarios[element("instance").value], baseService,
      Number(element("seed").value));
    element("visibility").value = "eligible";
    render();
  }

  function map(rows) {
    const canvas = element("map");
    const ctx = canvas.getContext("2d");
    const points = simulator.jobs.map(job => job.position).concat([simulator.scenario.depot]);
    const xs = points.map(point => point[0]);
    const ys = points.map(point => point[1]);
    const xmin = Math.min(...xs);
    const ymin = Math.min(...ys);
    const dx = Math.max(...xs) - xmin;
    const dy = Math.max(...ys) - ymin;
    const scale = Math.min((canvas.width - 60) / Math.max(dx, 1), (canvas.height - 60) / Math.max(dy, 1));
    const x0 = (canvas.width - scale * dx) / 2;
    const y0 = (canvas.height - scale * dy) / 2;
    const position = point => [x0 + (point[0] - xmin) * scale,
      canvas.height - y0 - (point[1] - ymin) * scale];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const {job, forecast} of rows) {
      const [x, y] = position(job.position);
      ctx.fillStyle = !forecast.reason ? "#098678" : job.status === "Completed" ? "#65888a" : "#bdcccc";
      ctx.beginPath();
      ctx.arc(x, y, forecast.reason ? 3 : 4.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    for (const nurse of simulator.nurses) {
      const current = nurse === simulator.current;
      if (!simulator.done && !current) continue;
      ctx.strokeStyle = current ? "#75219b" : "#517c7d";
      ctx.lineWidth = current ? 3.5 : 1.1;
      ctx.beginPath();
      nurse.trail.forEach((point, index) => {
        if (index === 0) ctx.moveTo(...position(point));
        else ctx.lineTo(...position(point));
      });
      ctx.stroke();
    }
    const [x, y] = position(simulator.scenario.depot);
    ctx.fillStyle = "#173536";
    ctx.fillRect(x - 5, y - 5, 10, 10);
    element("map-note").textContent = simulator.done
      ? "Lines show the nurses’ completed travel legs. The square marks the depot."
      : "Teal dots mark eligible visits. The purple line follows the current nurse. The square marks the depot.";
  }

  function render() {
    const current = simulator.current;
    const all = simulator.jobs.map(job => ({job, forecast: simulator.forecast(job)}));
    const eligible = all.filter(({forecast}) => !forecast.reason);
    const totals = [["Time (min)", minute(simulator.time)], ["Completed", simulator.completed],
      [simulator.done ? "Unserved" : "Remaining", simulator.jobs.length - simulator.completed],
      ["Travel (nurse-min)", minute(simulator.nurses.reduce((sum, nurse) => sum + nurse.travel, 0))],
      ["Waiting (nurse-min)", minute(simulator.nurses.reduce((sum, nurse) => sum + nurse.wait, 0))],
      [simulator.done ? "Final score" : "Score so far", minute(simulator.score)]];
    element("totals").innerHTML = totals.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
    element("decision-status").textContent = simulator.done
      ? `${simulator.horizonReached ? "Shift time reached" : "Shift finished"}. ${simulator.completed} of ${simulator.jobs.length} visits completed.`
      : `${nurseName(current.id)} · skill ${current.skill}. ${eligible.length} eligible ${eligible.length === 1 ? "visit" : "visits"}.`;
    element("defer").disabled = simulator.done;
    element("skip").disabled = simulator.done || eligible.length > 0;
    const visibility = element("visibility").value;
    const sort = element("sort").value;
    const value = ({job, forecast}) => sort === "deadline" ? job.window_end
      : sort === "id" ? job.id : forecast[sort] ?? Infinity;
    const displayed = (visibility === "eligible" ? eligible : visibility === "pending"
      ? all.filter(({job}) => job.status === "Pending") : all).slice();
    displayed.sort((a, b) => value(a) - value(b) || a.job.id - b.job.id);
    element("jobs").innerHTML = displayed.map(({job, forecast}) => {
      const status = forecast.reason ? `${forecast.reason}${job.nurse == null ? "" : ` · ${nurseName(job.nurse)}`}`
        : `<button type="button" data-visit="${job.id}" aria-label="Assign visit ${job.id}">Assign</button>`;
      return `<tr><td>${visit(job.id)}</td><td>${job.demand}</td><td>${minute(job.window_start)}–${minute(job.window_end)} min</td>` +
        `<td>${minute(forecast.travel)}</td><td>${minute(forecast.wait)}</td><td>${minute(forecast.service)}</td><td>${status}</td></tr>`;
    }).join("") || `<tr><td colspan="7">${simulator.done ? "See all visits to review outcomes." : "No visits match this view."}</td></tr>`;
    element("nurses").innerHTML = simulator.nurses.map(nurse =>
      `<tr${nurse === current ? ' class="current"' : ""}><td>${nurseName(nurse.id)}</td><td>${nurse.skill}</td>` +
      `<td>${nurse === current ? "Awaiting assignment" : nurse.status}</td><td>${visit(nurse.job)}</td><td>${nurse.completed}</td></tr>`).join("");
    element("events").innerHTML = simulator.events.slice(-24).reverse().map(event =>
      `<tr><td>${minute(event.time)}</td><td>${event.action}</td><td>${nurseName(event.nurse)}</td><td>${visit(event.job)}</td></tr>`).join("");
    map(all);
  }

  element("session").addEventListener("submit", event => { event.preventDefault(); start(); });
  element("visibility").addEventListener("change", render);
  element("sort").addEventListener("change", render);
  element("jobs").addEventListener("click", event => {
    const button = event.target.closest("button[data-visit]");
    if (button && simulator.assign(Number(button.dataset.visit))) {
      render();
      element("dispatch-title").focus();
    }
  });
  element("defer").addEventListener("click", () => { simulator.defer(); render(); });
  element("skip").addEventListener("click", () => {
    while (!simulator.done && !simulator.jobs.some(job => !simulator.forecast(job).reason)) simulator.defer();
    render();
    element("dispatch-title").focus();
  });
  start();
})();
