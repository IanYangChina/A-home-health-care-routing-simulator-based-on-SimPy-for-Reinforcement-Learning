# Architecture and research-model notes

## Event-driven model

- **Decision:** the environment exposes one idle nurse, and a policy selects an available appointment job or defers that worker. Decisions occur at SimPy event boundaries. Several nurses can simultaneously execute travel or service.
- **Job lifecycle:** pending (0) → assigned (1) → completed (2) / unserved (3). Jobs are never duplicated. A nurse may return to an elder more than once for distinct appointments.
- **Runtime:** one persistent `Process` and one `Store` per nurse; the scheduler uses `Event` wakeups to resume after completion. `Timeout` models travel/wait/service/return. No `Resource` or `PriorityResource` is necessary for the current problem; patients are passive entities.
- **Uncertain service:** a positive-clipped normal random draw is taken at service start, whose mean depends on skill, preference matrix and repeated visits. The planner sees expected mean only.
- **Termination:** all work completed, no more feasible work, or the shift horizon reached. At planned early completion nurses return to depot using SimPy timeouts.

## Research provenance

Source: Zhang, Liu, Yang, Chen and Huang (2023), C&IE 182, 109332, Figure 3 (original hierarchical MDP–CCP) and Table 1 (skill-demand service time). The article models (i) skill matching, (ii) mixed appointment windows with waiting before opening and no starts after closing, (iii) nurse–patient preferences and acquaintance, (iv) stochastic duration, (v) workload limits.

Differences made intentionally:

- **Planning-level skill-selection MDP removed.** This project's action selects a job for an idle nurse, and the simulation advances chronologically.
- **No chance-constrained programme inside `step()`.** A new ACO lookahead forecasts job routes using observable means; realised outcomes come from sampled durations.
- **No claim of reproducing reported metrics.** Older code uses numerical time accumulation and the `BWACO` implementation is not identical to the formal description. Here, SimPy processes represent actual nurse activity.
- **Lightweight reward:** +10 per completed service, -0.03 per travel minute, -0.05 per waiting minute, -8 for each unserved/failed job and -3 for invalid actions. These are demonstrator weights, **not the paper's objective**.
- **Time feasibility:** predicted start <= upper appointment window, waiting <= 40 minutes, predicted completion + estimated return to depot <= shift horizon. Actual service draws may violate this prediction.
- **No overpromised GPU support:** independent SimPy event loops can run on CPU workers. Batched CUDA stepping would require a separate event scheduling engine and equivalence tests; this package does not implement it.

## Future work

1. Improve assignment action to `(nurse_id, job_id)` with a single global scheduler, plus action masking.
2. Add an explicit release-time queue for dynamic new job arrivals and cancellations.
3. Extend comparisons across appointment layouts and independent training seeds.
4. Add SB3 learning curves and pruning to the comparative benchmark's Optuna studies.
5. Build domain-state snapshots at controlled decision boundaries for fork-based sampling. Rewind by replay, not generator cloning.
6. Consider a separate GPU vectorised backend only after validating CPU simulation semantics.
