"""SimPy's discrete-event execution core, independent of RL and route planners.

Every nurse owns a persistent SimPy process with a per-worker Store inbox.
Jobs are passive data. We use neither Resource nor PriorityResource: no shared
exclusive equipment exists in this scenario. Scheduling actions are external.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import erfc, exp, pi, sqrt
from typing import Any
import numpy as np
import simpy

from .data import Scenario, SERVICE_MEANS, travel_minutes


@dataclass
class Worker:
    id: int
    skill: int
    position: tuple[float, float, float]
    busy: bool = False
    total_travel: float = 0.
    total_wait: float = 0.
    total_service: float = 0.
    completed: int = 0
    visits: dict[int, int] = field(default_factory=dict)
    job_id: int | None = None
    arrival_at: float = 0.
    service_at: float = 0.
    expected_service: float = 0.
    return_at: float | None = None


class CareSimulation:
    """Authoritative chronological execution of independently active workers.

    Decision events are exposed by `current_worker` and `dispatch`/`defer`;
    a separate Gymnasium facade provides the standard RL-facing API.
    """

    def __init__(self, scenario: Scenario, *, seed: int = 0, duration_cv: float = .15,
                 common_random_numbers: bool = False, wait_penalty: float = .05):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        self.jobs = scenario.jobs
        self._positions = np.asarray([job.position for job in self.jobs], dtype=float)
        self._demand = np.asarray([job.demand for job in self.jobs], dtype=int)
        self._elder_ids = tuple(job.elder_id for job in self.jobs)
        self._elder_rows = np.asarray(self._elder_ids, dtype=int) - 1
        self._window_start = np.asarray([job.window_start for job in self.jobs], dtype=float)
        self._window_end = np.asarray([job.window_end for job in self.jobs], dtype=float)
        self._return_time = np.asarray([
            travel_minutes(job.position, scenario.depot, scenario.speed) for job in self.jobs
        ])
        # Index by job ID so policies share disturbances even if visits start in a different order.
        self._service_noise = (self.rng.standard_normal(len(self.jobs))
                               if common_random_numbers else None)
        self.workers = [Worker(i, s, scenario.depot) for i, s in enumerate(scenario.nurse_skills)]
        self.inboxes = [simpy.Store(self.env) for _ in self.workers]
        self.status = np.zeros(len(self.jobs), dtype=np.int8)  # 0 pending, 1 assigned, 2 done, 3 unserved
        self.ready: deque[int] = deque(range(len(self.workers)))
        self.parked: set[int] = set()
        self.active = 0
        self.duration_cv = duration_cv
        self.wait_penalty = wait_penalty
        self.events: list[dict[str, Any]] = []
        self.done = False
        self.truncated = False
        self.reward_buffer = 0.
        self._wake = self.env.event()
        self._returned = 0
        self._all_returned = self.env.event()
        self._horizon = self.env.timeout(float(scenario.horizon))
        self._processes = [self.env.process(self._worker_loop(worker)) for worker in self.workers]
        self._log('start', detail=f'{scenario.name}: {len(self.jobs)} jobs, {len(self.workers)} nurses')

    def _log(self, kind: str, *, nurse: int | None = None, job: int | None = None,
             detail: str = '', **extra: Any) -> None:
        ev = {'time': round(float(self.env.now), 5), 'type': kind,
              'nurse': nurse, 'job': job, 'detail': detail}
        ev.update(extra)
        self.events.append(ev)

    @property
    def current_worker(self) -> int | None:
        return self.ready[0] if self.ready and not self.done else None

    @property
    def pending(self) -> np.ndarray:
        return np.flatnonzero(self.status == 0)

    def expected_duration(self, nurse_id: int, job_id: int) -> float:
        worker, job = self.workers[nurse_id], self.jobs[job_id]
        base = SERVICE_MEANS[worker.skill - 1, job.demand - 1]
        pref = float(self.scenario.preferences[job.elder_id - 1, nurse_id])
        familiarity = .05 * worker.visits.get(job.elder_id, 0)
        return float(base * max(.65, pref - familiarity))

    def expected_durations(self, nurse_id: int) -> np.ndarray:
        """Forecast durations for a nurse, with zero for jobs outside their qualification."""
        nurse = self.workers[nurse_id]
        qualified = self._demand <= nurse.skill
        visits = np.fromiter((nurse.visits.get(elder, 0) for elder in self._elder_ids),
                             dtype=float, count=len(self.jobs))
        preferences = self.scenario.preferences[self._elder_rows[qualified], nurse_id]
        modifier = np.fmax(.65, preferences - .05 * visits[qualified])
        estimates = np.zeros(len(self.jobs))
        estimates[qualified] = SERVICE_MEANS[nurse.skill - 1, self._demand[qualified] - 1] * modifier
        return estimates

    def predict(self, nurse_id: int, job_id: int) -> dict[str, float]:
        worker, job = self.workers[nurse_id], self.jobs[job_id]
        travel = travel_minutes(worker.position, job.position, self.scenario.speed)
        arrival = float(self.env.now) + travel
        wait = max(0., job.window_start - arrival)
        start = arrival + wait
        service = self.expected_duration(nurse_id, job_id)
        end = start + service
        back = travel_minutes(job.position, self.scenario.depot, self.scenario.speed)
        return {'travel': travel, 'arrival': arrival, 'wait': wait,
                'start': start, 'end': end, 'return': back, 'service': service}

    def minutes_until_free(self, worker: Worker) -> float:
        """Forecast release using service still in progress and the public duration model."""
        now = float(self.env.now)
        if worker.return_at is not None:
            return max(0., worker.return_at - now)
        if worker.job_id is None:
            return 0.
        elapsed = now - worker.service_at
        mean = worker.expected_service
        spread = mean * self.duration_cv
        if elapsed < 1. or spread == 0.:
            return max(0., max(1., mean) - elapsed)
        # Condition the normal service model on the visit still running at this time.
        z = (elapsed - mean) / spread
        if z >= 35.:
            return spread / z
        return spread * (exp(-.5 * z * z) / (sqrt(pi / 2.) * erfc(z / sqrt(2.))) - z)

    def action_masks(self) -> np.ndarray:
        n = len(self.jobs)
        valid = np.zeros(n + 1, dtype=np.bool_)
        # 'defer' is legal, even with available jobs. It parks the nurse until
        # another worker completes or a future appointment window opens.
        valid[-1] = True
        nurse_id = self.current_worker
        if nurse_id is None:
            return valid
        nurse = self.workers[nurse_id]
        x, y, z = nurse.position
        job_x, job_y, job_z = self._positions.T
        # Apply the same floor-aware geometry as travel_minutes to every destination.
        horizontal = np.hypot(x - job_x, y - job_y)
        distance = np.where(horizontal < 1e-12, 9.6 * np.abs(z - job_z), horizontal + z + job_z)
        arrival = float(self.env.now) + distance / self.scenario.speed
        wait = np.maximum(0., self._window_start - arrival)
        start = arrival + wait
        end = start + self.expected_durations(nurse_id)
        valid[:-1] = ((self.status == 0) & (nurse.skill >= self._demand)
                      & (start <= self._window_end + 1e-8) & (wait <= self.scenario.max_wait + 1e-8)
                      & (end + self._return_time <= self.scenario.horizon + 1e-8))
        return valid

    def dispatch(self, job_id: int) -> bool:
        wid = self.current_worker
        if wid is None or job_id < 0 or job_id >= len(self.jobs) or not self.action_masks()[job_id]:
            return False
        self.ready.popleft()
        self.status[job_id] = 1
        worker = self.workers[wid]
        forecast = self.predict(wid, job_id)
        worker.job_id = job_id
        worker.arrival_at = forecast['arrival']
        worker.service_at = forecast['start']
        worker.expected_service = forecast['service']
        worker.busy = True
        self.active += 1
        self._log('dispatch', nurse=wid, job=job_id)
        self.inboxes[wid].put(job_id)
        return True

    def defer(self) -> None:
        wid = self.current_worker
        if wid is not None:
            self.ready.popleft()
            self.parked.add(wid)
            self._log('defer', nurse=wid)

    def _worker_loop(self, worker: Worker):
        while True:
            job_id = yield self.inboxes[worker.id].get()
            if job_id is None:
                break
            job = self.jobs[job_id]
            departure = worker.position
            travel = travel_minutes(departure, job.position, self.scenario.speed)
            self._log('travel_start', nurse=worker.id, job=job_id,
                      x0=departure[0], y0=departure[1], z0=departure[2],
                      x1=job.position[0], y1=job.position[1], z1=job.position[2])
            yield self.env.timeout(travel)
            worker.position = job.position
            worker.total_travel += travel
            self.reward_buffer -= .03 * travel
            self._log('arrive', nurse=worker.id, job=job_id)
            wait = max(0., job.window_start - float(self.env.now))
            if wait:
                self._log('wait_start', nurse=worker.id, job=job_id)
                yield self.env.timeout(wait)
                worker.total_wait += wait
                self.reward_buffer -= self.wait_penalty * wait
            if self.env.now > job.window_end + 1e-8:
                # Rare case: forecast was feasible, but a predecessor ran late.
                self.status[job_id] = 3
                self.reward_buffer -= 8.
                self._log('missed_window', nurse=worker.id, job=job_id)
            else:
                # Realise the duration using the nurse's expectation at service start.
                mean = self.expected_duration(worker.id, job_id)
                spread = mean * self.duration_cv
                sample = (self.rng.normal(mean, spread) if self._service_noise is None
                          else mean + spread * self._service_noise[job_id])
                duration = max(1.0, float(sample))
                self._log('service_start', nurse=worker.id, job=job_id, duration=duration)
                yield self.env.timeout(duration)
                worker.total_service += duration
                worker.completed += 1
                worker.visits[job.elder_id] = worker.visits.get(job.elder_id, 0) + 1
                self.status[job_id] = 2
                self.reward_buffer += 10.
                self._log('service_end', nurse=worker.id, job=job_id)
            worker.busy = False
            worker.job_id = None
            self.active -= 1
            self.ready.append(worker.id)
            if not self._wake.triggered:
                self._wake.succeed()
        # End-of-day return is executed by the worker process, not assumed.
        back = travel_minutes(worker.position, self.scenario.depot, self.scenario.speed)
        if back:
            worker.return_at = float(self.env.now) + back
            self._log('return_start', nurse=worker.id, x0=worker.position[0],
                      y0=worker.position[1], x1=self.scenario.depot[0], y1=self.scenario.depot[1])
            yield self.env.timeout(back)
            worker.total_travel += back
            self.reward_buffer -= .03 * back
            worker.position = self.scenario.depot
            worker.return_at = None
        self._log('return_end', nurse=worker.id)
        self._returned += 1
        if self._returned == len(self.workers) and not self._all_returned.triggered:
            self._all_returned.succeed()

    def _future_window_time(self) -> float | None:
        """Find a future release when everybody is idle/deferred.

        Since jobs can be dispatched before their TWB, waiting until TWB is a
        conservative progress rule, not an optimal dispatch policy.
        """
        now = float(self.env.now)
        future = [j.window_start for j, st in zip(self.jobs, self.status)
                  if st == 0 and j.window_start > now + 1e-7
                  and any(w.skill >= j.demand for w in self.workers)]
        return min(future) if future else None

    def _revive_parked(self):
        for w in sorted(self.parked):
            if w not in self.ready and not self.workers[w].busy:
                self.ready.append(w)
        self.parked.clear()

    def _finish(self, *, truncated: bool):
        if self.done:
            return
        self.truncated = truncated
        if not truncated and not self.active:
            for inbox in self.inboxes:
                inbox.put(None)
            # Nurse-process travel back to depot is part of the DES.
            if not self._all_returned.triggered:
                self.env.run(until=self._all_returned | self._horizon)
            if self.env.now >= self.scenario.horizon - 1e-8 and not self._all_returned.triggered:
                self.truncated = True
        unresolved = np.flatnonzero(self.status != 2)
        self.status[unresolved] = 3
        self.reward_buffer -= 8. * len(unresolved)
        self.done = True
        self._log('end', detail='horizon' if self.truncated else 'finished', unserved=len(unresolved))

    def advance(self):
        """Execute DES until the next agent scheduling decision or episode end."""
        while not self.done:
            now = float(self.env.now)
            if now >= self.scenario.horizon - 1e-8:
                self._finish(truncated=True)
                return
            if not np.any(self.status == 0) and self.active == 0:
                self._finish(truncated=False)
                return
            if self.ready and np.any(self.status == 0):
                return
            if self.active:
                self.env.run(until=self._wake | self._horizon)
                if self._wake.triggered:
                    self._wake = self.env.event()
                    self._revive_parked()
                continue
            next_window = self._future_window_time()
            if next_window is not None and next_window < self.scenario.horizon:
                self.env.run(until=self.env.timeout(next_window - now))
                self._revive_parked()
                # Pending jobs may all be permanently infeasible due to shift
                # limits; later windows still advance until final termination.
                continue
            self._finish(truncated=False)
            return

    def metrics(self) -> dict[str, Any]:
        c = int(np.sum(self.status == 2))
        return {
            'instance': self.scenario.name, 'time': round(float(self.env.now), 3),
            'jobs_total': len(self.jobs), 'jobs_completed': c,
            'jobs_unserved': int(np.sum(self.status == 3)),
            'jobs_pending': int(np.sum(self.status == 0)),
            'completion_rate': round(c / len(self.jobs), 4),
            'nurses': len(self.workers),
            'travel_minutes': round(sum(w.total_travel for w in self.workers), 3),
            'waiting_minutes': round(sum(w.total_wait for w in self.workers), 3),
            'service_minutes': round(sum(w.total_service for w in self.workers), 3),
            'truncated': self.truncated,
            'seeded_service_cv': self.duration_cv,
        }
