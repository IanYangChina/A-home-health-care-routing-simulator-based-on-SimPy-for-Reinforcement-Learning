# SHHCSRP · Home health care routing simulator

A SimPy and Gymnasium research simulator for assigning nurses to home visits in densely populated communities. It models nurse qualifications, appointment windows, stochastic service times, and nurse–patient preferences. It uses four benchmark instances from the [original SHHCSRP project](https://github.com/IanYangChina/SHHCSRP) to compare dispatch heuristics and ant colony optimisation with masked DQN and PPO. The simulator uses an event-driven formulation of the scheduling problem.

**Published paper.** Zhang, T., Liu, Y., Yang, X., Chen, J., & Huang, J. (2023). [Home health care routing and scheduling in densely populated communities considering complex human behaviours](https://doi.org/10.1016/j.cie.2023.109332). *Computers & Industrial Engineering*, 182, 109332.

[Open the results report and interactive simulator](https://ianyangchina.github.io/A-home-health-care-routing-simulator-based-on-SimPy-for-Reinforcement-Learning/)

## Examples

From the repository root with Python 3.10 or later:

```bash
# Compare the built-in dispatch policies on a small scenario.
python -m pip install -e .
shhcsrp-demo --instance A --limit 12 --compare --seed 7 --output output/demo

# Train a small MaskablePPO agent.
python -m pip install -e ".[rl]"
python -m shhcsrp_simpy.train_sb3 --instance A --limit 12 --steps 1024 --output-dir output/ppo
```
