"""
Loads trained models and regenerates the cross-phase evaluation table and
distance-over-time plot without re-running training.
"""
from project_pipeline import (
    Actor, DDPGAgent, PursuitEnv,
    evaluate_pair, print_eval, save_eval_table, plot_distance_over_time
)
import torch

env    = PursuitEnv()
N_EVAL = 50

def load_agent(path):
    agent = DDPGAgent()
    agent.actor.load_state_dict(torch.load(path, weights_only=True))
    agent.freeze()
    return agent

tracker0 = load_agent("models/tracker0_actor.pt")
tracker2 = load_agent("models/tracker2_actor.pt")
evader1  = load_agent("models/evader1_actor.pt")
evader3  = load_agent("models/evader3_actor.pt")

print(f"\n--- Cross-phase evaluation (N_EVAL={N_EVAL}) ---")
r_t0_straight = evaluate_pair(env, tracker0, None,    n_eval=N_EVAL)
r_t0_e1       = evaluate_pair(env, tracker0, evader1, n_eval=N_EVAL)
r_t0_e3       = evaluate_pair(env, tracker0, evader3, n_eval=N_EVAL)
r_t2_straight = evaluate_pair(env, tracker2, None,    n_eval=N_EVAL)
r_t2_e1       = evaluate_pair(env, tracker2, evader1, n_eval=N_EVAL)
r_t2_e3       = evaluate_pair(env, tracker2, evader3, n_eval=N_EVAL)

print_eval("tracker0 vs straight evader", r_t0_straight)
print_eval("tracker0 vs evader1",          r_t0_e1)
print_eval("tracker0 vs evader3",          r_t0_e3)
print_eval("tracker2 vs straight evader", r_t2_straight)
print_eval("tracker2 vs evader1",          r_t2_e1)
print_eval("tracker2 vs evader3",          r_t2_e3)

save_eval_table({
    "tracker0 vs straight evader": r_t0_straight,
    "tracker0 vs evader1":         r_t0_e1,
    "tracker0 vs evader3":         r_t0_e3,
    "tracker2 vs straight evader": r_t2_straight,
    "tracker2 vs evader1":         r_t2_e1,
    "tracker2 vs evader3":         r_t2_e3,
}, "figures/eval_table.png")

plot_distance_over_time({
    "Ph0 tracker vs Ph1 evader": r_t0_e1,
    "Ph2 tracker vs Ph1 evader": r_t2_e1,
    "Ph2 tracker vs Ph3 evader": r_t2_e3,
}, "figures/distance_over_time.png")

print("\nDone. figures/eval_table.png and figures/distance_over_time.png updated.")
