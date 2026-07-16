# 3D Co-evolutionary Pursuit-Evasion with ADHDP/DDPG

AE4350 Bio-inspired Intelligence and Learning for Aerospace Applications — TU Delft  
Duarte Fernandes

## How to run

```bash
pip install -r requirements.txt

# Full training (~2–3 h on CPU)
python project_pipeline.py

# Regenerate evaluation figures from saved models (fast)
python regen_eval_table.py

# Sensitivity analysis (~8 h on CPU)
python project_sensitivity.py
```

## Structure

```
project_pipeline.py       main pipeline: environment, DDPG agent, 4-phase co-evolutionary training
project_sensitivity.py    parameter sweep (evader speed, discount factor, capture radius)
regen_eval_table.py       reload saved models and regenerate evaluation figures

models/                   trained actor weights for all four phases
figures/                  all output figures and animations
sensitivity/              sweep results (results.json) and plots
dev_history/              numbered scripts showing the development progression
old_experiments/          earlier experiments (PPO baseline, intermediate versions)
```
