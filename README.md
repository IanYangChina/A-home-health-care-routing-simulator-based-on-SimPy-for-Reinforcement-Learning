# SHHCSRP — SimPy + Gymnasium routing & scheduling demo

[Interactive Simulator and Project Reports](https://ianyangchina.github.io/A-home-health-care-routing-simulator-based-on-SimPy-for-Reinforcement-Learning/)

A **new, event-driven** implementation of the stochastic home-health-care scheduling and routing problem, grounded in the data and concepts of:

> Zhang et al., *Home health care routing and scheduling in densely populated communities considering complex human behaviours*, Computers & Industrial Engineering 182 (2023), 109332. https://doi.org/10.1016/j.cie.2023.109332

And the associated [SHHCSRP](https://github.com/IanYangChina/SHHCSRP) code by Xintong Yang (original core scripts retained in `legacy/` for provenance).

**This is a new formulation with event-driven dispatch decisions, not a reproduction of the paper's original nested QL–BWACO experimental results.** It is a teaching/application demonstration, not a clinically validated system.

## What works

- **SimPy:** each nurse runs a persistent process using a `simpy.Store` inbox; travelling, waiting, service, and return journeys consume simulated time. Nurses execute concurrently.
- **Real original inputs:** original Excel instances A/B/C/D (60/60/159/303 appointments) and stored nurse–elder preference matrices. Repeated elder IDs represent multiple appointments. `JobNum` in the workbook is **not** used as a unique job ID; Python indexes are stable identifiers.
- **Scheduling constraints:** nurse qualification vs job skill level, appointment time windows, wait limit, finite work horizon including forecast return to depot, nurse–elder preferences and acquaintance effects.
- **Stochasticity:** service durations sampled only when service starts. The scheduler sees expected durations (normal distribution, clipped at 1 minute); the scenario uses the paper's mean duration table and preference rule.
- **Gymnasium:** `reset()`, `step()`, observations (`Dict` of fixed-shape `Box` arrays), `Discrete(n_jobs + 1)` actions, termination/truncation, `info` and feasibility masks.
- **Algorithms:** random, greedy, and ant colony lookahead are available in the demo CLI. The comparative benchmark also includes an earliest-deadline baseline.
- **RL & tuning:** the comparative benchmark tunes masked SB3 DQN and `sb3-contrib` MaskablePPO with Optuna. A smaller MaskablePPO example is also available.
- **Reporting:** a **self-contained interactive HTML** with route map, per-nurse timeline, metrics and filterable event trace, plus CSV/JSON.
- **Parallelism:** optional CPU multiprocessing of independent episodes. **No GPU-native DES backend is included.** Standard SimPy event queues cannot be accelerated merely by sending data to CUDA; an independently validated batched GPU simulator is a separate future project.

## Installation

Python **3.10+** and a package index reachable for dependencies are required:

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\\Scripts\\activate
python -m pip install -e .
# optional RL and hyperparameter search:
python -m pip install -e '.[rl]'
# optional tests:
python -m pip install -e '.[test]'
```

If installed as a normal wheel, the data files are included inside the package. To use other datasets, supply `--data-dir /path/to/data` to `shhcsrp-demo`.

## Quickstart

Run from repository root:

```bash
# A lightweight scenario using 35 appointments from the original A Excel workbook
shhcsrp-demo --instance A --limit 35 --policy aco --seed 7 --output output/aco
# Compare all baselines under the same scenario and random seed
shhcsrp-demo --instance A --limit 35 --compare --seed 7 --output output/compare
# Full original instances: --limit 0 (more computationally expensive)
shhcsrp-demo --instance B --limit 0 --policy greedy --output output/full_B
```

Open `output/aco.html` in a browser; **no web server is needed**. Files also include `output/aco.json` and `output/aco_events.csv`.

### Custom Gymnasium agent

```python
from shhcsrp_simpy import SHHCSRPGymEnv

env = SHHCSRPGymEnv(instance='A', job_limit=35)
obs, info = env.reset(seed=42)
while True:
    mask = info['action_mask']                 # numpy bool[n_jobs+1]
    action = int(next(i for i, ok in enumerate(mask) if ok))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(info['metrics'])
        break
```

**Action** `0..n_jobs-1` assigns the *currently selected idle nurse* to that job. `n_jobs` means `defer`: park the nurse until another worker becomes idle or a future time window opens. Every action has a mask; use `MaskablePPO` for safe reinforcement-learning exploration. Invalid actions are penalised and treated as deferrals.

`obs['jobs']` contains one row per appointment: `[x, y, z, window_begin, window_end, skill, status, expected_service, elder_id]` (normalised). `obs['worker']` contains the selected nurse's location, skill, index, current time, jobs completed, and time left. `obs['global']` contains current time, fraction completed, fraction pending and proportion busy.

### RL and Optuna (optional)

```bash
python -m shhcsrp_simpy.train_sb3 --instance A --limit 35 --steps 10000 --output-dir output/rl
python -m shhcsrp_simpy.tune_optuna --instance A --limit 35 --trials 10 --steps 4000
```

This RL implementation intentionally uses an established algorithm, `MaskablePPO`, rather than creating a custom learner. The Optuna example uses fixed validation seeds, which must be kept distinct from final held-out evaluations. Small example budgets demonstrate the pipeline, **not** high-quality policy convergence.

### Comparative benchmark and interactive report

Open the [project site](docs/report/index.html) from a local checkout. The homepage presents the latest measured results on all four appointment layouts. The other main pages cover the [algorithm setup](docs/report/setup/index.html), the [development history](docs/report/development/index.html), and the [playable dispatch simulator](docs/report/simulator/index.html). The [tuned benchmark](docs/report/benchmark/index.html) and individual studies retain their interactive charts and route explorers.

To reproduce the benchmark in a Python 3.11 conda environment with the CUDA build used for this report:

```bash
conda create -n shhcsrp python=3.11.11 pip
conda activate shhcsrp
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[rl,test]' stable-baselines3==2.6.0 sb3-contrib==2.6.0 numpy==2.2.3 'optuna>=4,<5'
python -m shhcsrp_simpy.benchmark --device cpu
```

The default experiment uses all 60 appointments in instance A, eight Optuna trials per algorithm at 4,096 steps each, and three independent 24,576-step final runs per algorithm. The output directory contains seeded checkpoints, one representative checkpoint per agent, held-out episode metrics, and the trace and training profiles in JSON. Regenerate the detail report from saved results with `python -m shhcsrp_simpy.benchmark --report-only output/benchmark/benchmark.json`.

Optuna searches two equal hidden layers of 64, 128, or 256 units and batch sizes of 32, 64, or 128 for both agents. It searches DQN target updates every 256, 512, 1,024, or 2,048 steps and PPO rollouts of 128, 256, or 512 steps. Learning rate, discount factor, DQN exploration decay, and PPO entropy remain tunable. Use `--network` to fix the hidden layers or `--fixed-params` to reuse a previous study. SimPy runs on the CPU. The published experiment uses CPU training for A, B, and C and uses CUDA for DQN and CPU for PPO on D.

```powershell
foreach ($instance in 'A', 'B', 'C') {
    $label = $instance.ToLower()
    python -m shhcsrp_simpy.benchmark --instance $instance --device cpu --output-dir "output/tuned-$label" --report "docs/report/$label.html"
}
python -m shhcsrp_simpy.benchmark --instance D --device cuda --ppo-device cpu --output-dir output/tuned-d --report docs/report/d.html
```

The fixed policies are random, greedy, earliest deadline, and ant colony. For DQN and PPO, action `j` refers to appointment `j` throughout the shift. The final action always defers the current nurse. The eligibility mask changes as the shift advances. DQN applies it during exploration, greedy action selection, and the replay target calculation. MaskablePPO applies it during training and evaluation.

Coordinates, time, and counts are scaled by fixed denominators and clipped to `[0, 1]`. The eligibility flags are binary. There is no running-statistics observation or reward normalization. Optuna and the final RL selection use five validation seeds. Each method's best-validation checkpoint supplies its 12 held-out episode traces. Training diagnostics average each seed's loss samples within 512-step windows, then plot the mean and sample standard deviation across training seeds. The report also lists the completed training episodes per seed.

After uploading the repository to GitHub, choose **GitHub Actions** under **Settings → Pages → Build and deployment → Source**. Run **Publish benchmark report** from the Actions tab on the default branch for the first deployment. The workflow publishes `docs/report/` and shows the site URL on the deployment job. Subsequent changes to that folder on the default branch trigger deployment. The simulator runs in each visitor’s browser, where they can assign or defer nurses, inspect outcomes, and reset the shift. All runtime assets are shipped with the site. [GitHub Pages setup](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

### PI-led development and guided exploration

The [development timeline](docs/report/development/index.html) links each round's report and records ny's feedback, agent and subagent work, computation, and the logged model configuration. Round reports distinguish measured artifacts from summaries recovered from the conversation.

The [guided-exploration report](docs/report/exploration/index.html) compares matched controls, short-wait guidance, and bounded ant-colony guidance on A and D. The DQN variants mix guidance into epsilon exploration so replay contains the simulator transitions those decisions produce. PPO's guided variants imitate demonstrations from four separate seeds, then train on their own on-policy rollouts. Each fit uses 12,288 decision steps with the preceding tuned hyperparameters and original reward weights.

The report shows completed and unserved visits, travel and waiting minutes, and the weighted contributions and total. Every 512 steps it averages the most recent 20 completed episodes separately for each training seed, then displays the mean and sample SD across seeds. Held-out summaries include all three trained checkpoints. Route explorers show the predeclared training seed 11.

Run `python -m shhcsrp_simpy.exploration --instance A --family dqn` once for each instance (`A`, `D`) and family (`dqn`, `ppo`). The reported DQN run on D adds `--device cuda`. Then run `python -m shhcsrp_simpy.exploration --render A D`. The earlier tuned results under `output/tuned-a` and `output/tuned-d` supply the fixed settings and baseline traces. New checkpoints, episode-level reward terms, and profiles are written under `output/rounds/06-guided-exploration/`.

The [waiting-cost and return-length study](docs/report/returns/index.html) compares matched C and D controls against wait costs of 0.15 and 0.30 points per nurse-minute and longer return estimates. DQN uses 16- or 32-step masked replay with the prior tuned discount. PPO uses 512-step rollouts throughout and tests `gae_lambda=1` both at the tuned discount and at `gamma=1`. PPO still bootstraps at rollout boundaries inside a shift. The modeled shift end is terminal for learning in all arms. Held-out scores use the original reward weights so the arms share a reference scale.

Run `python -m shhcsrp_simpy.returns_experiment --instance C --family dqn` once per instance (`C`, `D`) and family (`dqn`, `ppo`). The reported DQN run on D adds `--device cuda`. Then run `python -m shhcsrp_simpy.returns_experiment --render C D`. Checkpoints and per-episode rewards are saved under `output/rounds/07-wait-and-returns/`.

The [combined exploration and longer-returns study](docs/report/guided-returns/index.html) trains both agents on A–D. Every learned arm charges 0.15 points per nurse-minute of waiting and uses longer returns. Within each layout and family, the exploration control, short-wait guide, and ACO guide share the same settings. DQN uses 32 steps on C and 16 elsewhere. PPO uses `gae_lambda=1` with 512-step rollouts and the tuned discounts.

Run `python -m shhcsrp_simpy.guided_returns --instance A --family dqn` for each instance and family, adding `--device cuda` for DQN on D to reproduce the reported device assignment. Run `python -m shhcsrp_simpy.guided_returns --render A B C D` to build the report from `output/rounds/08-guided-long-returns/`.

Run `python -m shhcsrp_simpy.site_results` to regenerate the latest-results homepage from the saved Round 8 summary. Run `python -m shhcsrp_simpy.browser_data` to export the bundled scenario data used by the browser simulator.

### CPU multiprocessing

```bash
python -m shhcsrp_simpy.parallel --instance A --limit 35 --policy greedy --workers 4 --seeds 12
```

## Architectural mapping

| Original / domain object | New code | SimPy primitive |
|---|---|---|
| Nurse | `Worker`, `CareSimulation._worker_loop` | `Process` waits on a per-nurse `Store` |
| Patient/elder | Identifier and location on `Job` | Passive data, **not** `Resource` |
| Appointment demand | `Job` | Passive data, tracked in `status` |
| Travel, wait, service | `_worker_loop` | `Timeout` |
| Assignment | `dispatch()` | `Store.put(job_id)` |
| Completion decision notification | `_wake` | `Event` |
| Shared priority queue/resource | None in source model | `Resource`/`PriorityResource` **not needed** |
| RL policy | `SHHCSRPGymEnv` + SB3 | External to SimPy |
| Classical optimiser | `ACOPolicy` | External to SimPy |

See `docs/architecture.md` for details on design decisions and paper/code differences.

## Demonstrator limitations and distinctions

1. This implementation uses **actual sampled service durations**; the legacy solver mostly operated on expected durations and chance-constraint estimates. It therefore cannot match the published numeric results exactly, by design.
2. It uses a **one-nurse/one-job event-driven dispatch MDP** rather than the original skill-level nurse-selection MDP plus an inner CCP optimiser.
3. Simulated travel time follows the original `get_data` distance calculation (including its unusual vertical-distance rule), with original walking speed = 60 distance units/minute. This is explicitly a model assumption.
4. It uses an approximate expected-duration feasibility check for the 480-minute horizon, so stochastic realisations may overrun. Missing deadlines and unfinished work are penalised; this is a **simulation outcome**, not a chance-constrained guarantee.
5. ACO is an agent acting through observations/masks, not an environment, and its lookahead is a shallow expected-time approximation rather than the published BWACO method. ACO and the baseline heuristic both use the Gymnasium interface.
6. Default worker skills are selected demo configurations; the archive includes several choices in its experimental configurations. Alter `load_scenario(nurse_skills=...)` if necessary; preference matrices must have corresponding columns.
7. The prototype has **no fork/rewind/checkpoint API**. Those are optional features that require a carefully designed state reconstruction layer because live SimPy generators cannot simply be deep-copied.
8. HTML routes use XY views of the original spatial coordinates, with z retained in event/job data; there is no animated live worker map.

## Validation and examples

The comparative report records runs with SimPy, Gymnasium, Stable-Baselines3, and Optuna. The pre-rendered [ACO preview](examples/demo_aco.html) is from the earlier API-shim smoke runner. Run the demo CLI to regenerate an individual episode report with the installed libraries.

## Testing

```bash
python -m pytest -q
python -m compileall -q src tests
```

A basic baseline correctness test checks that completed jobs never repeat, nurses do not serve skills above qualification, time progresses monotonically, identical seeds reproduce event traces, and Gymnasium's observation/mask spaces are well formed. See `tests/`.

## Reproducibility / fair comparison

- Supply the same instance, seed, nurse skills and service-time parameters for each agent.
- All policies use only observations and validity masks, not private future service-time draws.
- The single-episode demo uses a sequential random stream. The comparative benchmark uses per-job disturbances so the same appointment receives the same random draw across methods given an episode seed.
- Compare distributions over multiple seeds, not single wins. Report achieved completion, waiting, travel and service time, reward and wall-clock runtime.

## Authors / attribution

This demonstration reuses source dataset files and references original code from Xintong Yang's [SHHCSRP](https://github.com/IanYangChina/SHHCSRP).
