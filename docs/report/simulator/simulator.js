"use strict";

(() => {
  const element = id => document.getElementById(id);
  const {scenarios, baseService} = globalThis.DISPATCH_DATA;
  const minute = value => value == null ? "—" : value.toFixed(1);
  const visit = id => id == null ? "—" : `#${id}`;
  const nurseName = id => id == null ? "—" : `Nurse ${id + 1}`;
  const map = new globalThis.DispatchMap3D(element("map"), element("map-status"), element("map-clock"),
    element("map-fallback"), element("map-reset"));
  let simulator;

  function start() {
    simulator = new globalThis.DispatchSimulator(scenarios[element("instance").value], baseService,
      Number(element("seed").value));
    map.reset(simulator);
    element("visibility").value = "eligible";
    render();
  }

  function render(assignment) {
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
      const location = map.location(job.position);
      const status = forecast.reason ? `${forecast.reason}${job.nurse == null ? "" : ` · ${nurseName(job.nurse)}`}`
        : `<button type="button" data-visit="${job.id}" aria-label="Assign visit ${job.id} at ${location}">Assign</button>`;
      return `<tr><td>${visit(job.id)}<span class="sim-location">${location}</span></td><td>${status}</td><td>${job.demand}</td>` +
        `<td>${minute(job.window_start)}–${minute(job.window_end)} min</td><td>${minute(forecast.travel)}</td>` +
        `<td>${minute(forecast.wait)}</td><td>${minute(forecast.service)}</td></tr>`;
    }).join("") || `<tr><td colspan="7">${simulator.done ? "See all visits to review outcomes." : "No visits match this view."}</td></tr>`;
    element("nurses").innerHTML = simulator.nurses.map(nurse =>
      `<tr${nurse === current ? ' class="current"' : ""}><td>${nurseName(nurse.id)}</td><td>${nurse.skill}</td>` +
      `<td>${nurse === current ? "Awaiting assignment" : nurse.status}</td><td>${visit(nurse.job)}</td><td>${nurse.completed}</td></tr>`).join("");
    element("events").innerHTML = simulator.events.slice(-24).reverse().map(event =>
      `<tr><td>${minute(event.time)}</td><td>${event.action}</td><td>${nurseName(event.nurse)}</td><td>${visit(event.job)}</td></tr>`).join("");
    map.update(all, assignment);
  }

  element("session").addEventListener("submit", event => { event.preventDefault(); start(); });
  element("visibility").addEventListener("change", () => render());
  element("sort").addEventListener("change", () => render());
  element("jobs").addEventListener("click", event => {
    const button = event.target.closest("button[data-visit]");
    if (!button) return;
    const job = simulator.jobs[Number(button.dataset.visit)];
    const nurse = simulator.current;
    const legIndex = nurse.legs.length;
    if (simulator.assign(job.id)) {
      render({nurse, job, leg: nurse.legs[legIndex]});
      element("dispatch-title").focus({preventScroll: true});
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
