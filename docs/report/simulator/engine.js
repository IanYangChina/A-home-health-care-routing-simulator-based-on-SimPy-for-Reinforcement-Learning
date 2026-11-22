"use strict";

(() => {
  const epsilon = 1e-8;

  function serviceNoise(seed, size) {
    let state = seed >>> 0;
    function uniform() {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      return (state + 0.5) / 4294967296;
    }
    return Array.from({length: size}, () =>
      Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()));
  }

  class DispatchSimulator {
    constructor(scenario, baseService, seed, draws = null) {
      this.scenario = scenario;
      this.baseService = baseService;
      this.noise = draws ?? serviceNoise(seed, scenario.jobs.length);
      this.jobs = scenario.jobs.map(job => ({...job, status: "Pending", nurse: null}));
      this.nurses = scenario.nurses.map((skill, id) => ({id, skill, position: scenario.depot,
        status: "Ready", job: null, completed: 0, visits: new Map(), legs: [],
        travel: 0, wait: 0, service: 0}));
      this.ready = this.nurses.slice();
      this.queue = [];
      this.events = [];
      this.time = 0;
      this.decisions = 0;
      this.done = false;
      this.horizonReached = false;
      this.score = 0;
      this.log("Shift started");
    }

    get current() { return this.ready[0]; }

    log(action, nurse = null, job = null) {
      this.events.push({time: this.time, action, nurse: nurse?.id ?? null, job: job?.id ?? null});
    }

    travel(a, b) {
      const horizontal = Math.hypot(a[0] - b[0], a[1] - b[1]);
      const distance = horizontal < 1e-12 ? 9.6 * Math.abs(a[2] - b[2]) : horizontal + a[2] + b[2];
      return distance / this.scenario.speed;
    }

    expected(nurse, job) {
      const base = this.baseService[nurse.skill - 1][job.demand - 1];
      const preference = this.scenario.preferences[job.elder_id - 1][nurse.id];
      const familiarity = 0.05 * (nurse.visits.get(job.elder_id) ?? 0);
      return base * Math.max(0.65, preference - familiarity);
    }

    forecast(job) {
      const nurse = this.current;
      if (job.status !== "Pending") return {reason: job.status};
      if (!nurse || this.done) return {reason: "Shift ended"};
      if (nurse.skill < job.demand) return {reason: "Skill requirement"};
      const travel = this.travel(nurse.position, job.position);
      const arrival = this.time + travel;
      const wait = Math.max(0, job.window_start - arrival);
      const start = arrival + wait;
      const service = this.expected(nurse, job);
      const back = start + service + this.travel(job.position, this.scenario.depot);
      let reason = "";
      if (start > job.window_end + epsilon) reason = "Window closes before arrival";
      else if (wait > this.scenario.maxWait + epsilon) reason = "Wait exceeds limit";
      else if (back > this.scenario.horizon + epsilon) reason = "Return exceeds shift";
      return {reason, travel, wait, start, service, back};
    }

    book(nurse, phase, minutes, job = null) {
      nurse.status = phase;
      const end = this.time + minutes;
      if (phase === "Travelling" || phase === "Returning") {
        nurse.legs.push({start: this.time, end, from: nurse.position,
          to: job ? job.position : this.scenario.depot});
      }
      this.queue.push({time: end, nurse, phase, minutes, job});
    }

    assign(id) {
      const job = this.jobs[id];
      if (!job || this.forecast(job).reason) return false;
      const nurse = this.ready.shift();
      job.status = "Assigned";
      job.nurse = nurse.id;
      nurse.job = job.id;
      this.decisions++;
      this.log("Assigned", nurse, job);
      this.book(nurse, "Travelling", this.travel(nurse.position, job.position), job);
      this.advance();
      return true;
    }

    defer() {
      if (!this.current || this.done) return;
      const nurse = this.ready.shift();
      nurse.status = "Deferred";
      this.decisions++;
      this.log("Deferred", nurse);
      this.advance();
    }

    service(nurse, job) {
      if (this.time > job.window_end + epsilon) {
        job.status = "Unserved";
        this.score -= 8;
        this.log("Missed window", nurse, job);
        this.release(nurse);
        return;
      }
      const expected = this.expected(nurse, job);
      const duration = Math.max(1, expected + expected * 0.15 * this.noise[job.id]);
      this.log("Service started", nurse, job);
      this.book(nurse, "In service", duration, job);
    }

    release(nurse) {
      nurse.status = "Ready";
      nurse.job = null;
      this.ready.push(nurse);
    }

    process({nurse, phase, minutes, job}) {
      if (phase === "Travelling") {
        nurse.position = job.position;
        nurse.travel += minutes;
        this.score -= 0.03 * minutes;
        this.log("Arrived", nurse, job);
        const wait = Math.max(0, job.window_start - this.time);
        if (wait) {
          this.log("Waiting for window", nurse, job);
          this.book(nurse, "Waiting", wait, job);
        } else this.service(nurse, job);
      } else if (phase === "Waiting") {
        nurse.wait += minutes;
        this.score -= 0.15 * minutes;
        this.service(nurse, job);
      } else if (phase === "In service") {
        nurse.service += minutes;
        nurse.completed++;
        nurse.visits.set(job.elder_id, (nurse.visits.get(job.elder_id) ?? 0) + 1);
        job.status = "Completed";
        this.score += 10;
        this.log("Visit completed", nurse, job);
        this.release(nurse);
      } else {
        nurse.travel += minutes;
        nurse.position = this.scenario.depot;
        nurse.status = "At depot";
        this.score -= 0.03 * minutes;
        this.log("Returned to depot", nurse);
      }
    }

    nextEvents() {
      this.queue.sort((a, b) => a.time - b.time);
      this.time = Math.min(this.scenario.horizon, this.queue[0].time);
      while (this.queue.length && this.queue[0].time <= this.time) {
        this.process(this.queue.shift());
        this.queue.sort((a, b) => a.time - b.time);
      }
    }

    revive() {
      for (const nurse of this.nurses) if (nurse.status === "Deferred") this.release(nurse);
    }

    finish() {
      this.horizonReached = this.time >= this.scenario.horizon - epsilon;
      if (!this.horizonReached) {
        for (const nurse of this.nurses) {
          const back = this.travel(nurse.position, this.scenario.depot);
          if (back) {
            this.book(nurse, "Returning", back);
            this.log("Returning to depot", nurse);
          } else nurse.status = "At depot";
        }
        while (this.queue.length && this.time < this.scenario.horizon) this.nextEvents();
        this.horizonReached = this.queue.length > 0;
      }
      for (const job of this.jobs) {
        if (job.status === "Completed") continue;
        this.score -= 8;
        job.status = "Unserved";
      }
      if (this.horizonReached) {
        for (const nurse of this.nurses) if (nurse.status !== "At depot") nurse.status = "Shift ended";
      }
      this.done = true;
      this.ready.length = 0;
      this.log(this.horizonReached ? "Shift time reached" : "Shift finished");
    }

    advance() {
      while (this.time < this.scenario.horizon - epsilon) {
        const pending = this.jobs.filter(job => job.status === "Pending");
        if (pending.length && this.ready.length) return;
        if (this.queue.length) {
          this.nextEvents();
          if (this.ready.length) this.revive();
          continue;
        }
        const maxSkill = Math.max(...this.scenario.nurses);
        const opens = pending.filter(job => job.window_start > this.time + 1e-7 && job.demand <= maxSkill)
          .map(job => job.window_start);
        const next = Math.min(...opens);
        if (next >= this.scenario.horizon) break;
        this.time = next;
        this.revive();
      }
      this.finish();
    }

    get completed() { return this.jobs.filter(job => job.status === "Completed").length; }
  }

  globalThis.DispatchSimulator = DispatchSimulator;
})();
