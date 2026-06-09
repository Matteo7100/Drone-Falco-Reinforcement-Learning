"""
eval_quad.py — rigorous KPI evaluation for a trained quadcopter PPO policy.

Run:
    python eval_quad.py --model quad_model_best.pth --episodes 20

Outputs:
    eval_results.json   (load this in the dashboard)

KPIs collected
--------------
Per-episode
  • total_reward          — sum of rewards over the episode
  • episode_length        — steps survived (max = 1500)
  • termination           — 'timeout' | 'flip' | 'out_of_bounds'
  • hover_time_s          — seconds spent within 0.3 m of target (sim time)
  • settling_time_s       — first time the drone enters and STAYS within 0.3 m
                           for at least 1 s; None if never achieved
  • overshoot_m           — max distance from target *after* first reaching 0.3 m
  • steady_state_error_m  — mean distance from target during last 5 s of episode
  • max_tilt_deg          — maximum roll or pitch angle seen during episode
  • mean_tilt_deg         — mean |tilt| during episode
  • mean_speed_ms         — mean linear speed [m/s]
  • mean_ang_speed_rads   — mean angular speed [rad/s]
  • ctrl_effort           — mean squared action (measures energy use)
  • dist_to_target        — distance trajectory (full timeseries, per episode)
  • pos_trajectory        — [x,y,z] trajectory (full timeseries, per episode)

Aggregate (across all episodes)
  • success_rate          — fraction of episodes ending in 'timeout' (survived)
  • mean / std of every scalar KPI above
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from quad_env import QuadcopterEnv


# =============================================================================
#  NETWORK — must match training architecture exactly
# =============================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        def mlp(i, o):
            return nn.Sequential(
                nn.Linear(i, 256), nn.Tanh(),
                nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, o)
            )

        self.actor  = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, x):
        mu    = self.actor(x)
        value = self.critic(x).squeeze(-1)
        std   = torch.exp(torch.clamp(self.log_std, -3.0, 0.5))
        return mu, std, value

    def act(self, obs_t):
        mu, _, _ = self.forward(obs_t)
        return mu   # deterministic at test time


# =============================================================================
#  EPISODE RUNNER
# =============================================================================
def run_episode(env, model, device, hover_threshold=0.3, hover_hold_s=1.0, Ts=0.01):
    obs  = env.reset()
    done = False

    # --- raw timeseries ---
    dists       = []   # distance to target at each step
    positions   = []   # [x,y,z]
    eulers      = []   # [phi, theta, psi]
    speeds      = []   # linear speed
    ang_speeds  = []   # angular speed
    actions     = []   # raw actions in [-1,1]
    rewards     = []

    termination = "timeout"

    while not done:
        obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.act(obs_t).cpu().numpy()

        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action.copy())

        # un-normalise observation to recover physical quantities
        # obs = [pos/10, euler/pi, vel/5, ang_vel/5]
        pos     = obs[0:3] * 10.0
        euler   = obs[3:6] * np.pi
        vel     = obs[6:9] * 5.0
        ang_vel = obs[9:12] * 5.0

        dist = float(np.linalg.norm(env.target_position - pos))
        dists.append(dist)
        positions.append(pos.tolist())
        eulers.append(euler.tolist())
        speeds.append(float(np.linalg.norm(vel)))
        ang_speeds.append(float(np.linalg.norm(ang_vel)))

        if done and 'termination' in info:
            termination = info['termination']

    # ── Compute KPIs from timeseries ─────────────────────────────────────────
    dists      = np.array(dists)
    eulers_arr = np.array(eulers)
    n          = len(dists)

    # 1. Hover time: steps within threshold
    in_hover      = dists < hover_threshold
    hover_steps   = int(np.sum(in_hover))
    hover_time_s  = hover_steps * Ts

    # 2. Settling time: first moment drone enters threshold and STAYS for hold_steps
    hold_steps    = int(hover_hold_s / Ts)
    settling_time = None
    for i in range(n - hold_steps):
        if np.all(in_hover[i: i + hold_steps]):
            settling_time = i * Ts
            break

    # 3. Overshoot: max distance AFTER first reaching threshold
    first_reach = None
    for i, d in enumerate(dists):
        if d < hover_threshold:
            first_reach = i
            break
    overshoot = float(np.max(dists[first_reach:])) if first_reach is not None else float(np.max(dists))

    # 4. Steady-state error: mean distance in last 5 s
    last_n = min(int(5.0 / Ts), n)
    steady_state_error = float(np.mean(dists[-last_n:]))

    # 5. Tilt
    tilts = np.max(np.abs(eulers_arr[:, :2]), axis=1)   # max of |roll|, |pitch|
    max_tilt_deg  = float(np.degrees(np.max(tilts)))
    mean_tilt_deg = float(np.degrees(np.mean(tilts)))

    # 6. Control effort
    actions_arr = np.array(actions)
    ctrl_effort = float(np.mean(np.sum(actions_arr ** 2, axis=1)))

    return {
        "total_reward":        float(np.sum(rewards)),
        "episode_length":      n,
        "termination":         termination,
        "hover_time_s":        hover_time_s,
        "settling_time_s":     settling_time,
        "overshoot_m":         overshoot,
        "steady_state_error_m": steady_state_error,
        "max_tilt_deg":        max_tilt_deg,
        "mean_tilt_deg":       mean_tilt_deg,
        "mean_speed_ms":       float(np.mean(speeds)),
        "mean_ang_speed_rads": float(np.mean(ang_speeds)),
        "ctrl_effort":         ctrl_effort,
        "dist_to_target":      dists.tolist(),
        "pos_trajectory":      positions,
    }


# =============================================================================
#  AGGREGATE
# =============================================================================
def aggregate(episodes):
    scalar_keys = [
        "total_reward", "episode_length", "hover_time_s",
        "steady_state_error_m", "max_tilt_deg", "mean_tilt_deg",
        "mean_speed_ms", "mean_ang_speed_rads", "ctrl_effort", "overshoot_m"
    ]

    agg = {}
    for k in scalar_keys:
        vals = [ep[k] for ep in episodes]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"]  = float(np.std(vals))
        agg[f"{k}_min"]  = float(np.min(vals))
        agg[f"{k}_max"]  = float(np.max(vals))

    # settling time (can be None)
    settling = [ep["settling_time_s"] for ep in episodes if ep["settling_time_s"] is not None]
    agg["settling_time_s_mean"]    = float(np.mean(settling))   if settling else None
    agg["settling_time_s_success"] = len(settling) / len(episodes)

    # termination breakdown
    terms = [ep["termination"] for ep in episodes]
    agg["success_rate"]       = terms.count("timeout")       / len(episodes)
    agg["flip_rate"]          = terms.count("flip")          / len(episodes)
    agg["out_of_bounds_rate"] = terms.count("out_of_bounds") / len(episodes)

    return agg


# =============================================================================
#  MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="quad_model_best.pth")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out",      type=str, default="eval_results.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    env    = QuadcopterEnv(render=False)   # no GUI for speed

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded '{args.model}'")

    episodes = []
    for i in range(1, args.episodes + 1):
        ep = run_episode(env, model, device)
        episodes.append(ep)
        print(
            f"  Episode {i:3d}/{args.episodes} | "
            f"term={ep['termination']:15s} | "
            f"hover={ep['hover_time_s']:.1f}s | "
            f"ss_err={ep['steady_state_error_m']:.3f}m | "
            f"reward={ep['total_reward']:.1f}"
        )

    env.close()

    agg = aggregate(episodes)

    print("\n── Aggregate KPIs ──────────────────────────────────────")
    print(f"  Success rate        : {agg['success_rate']*100:.1f}%")
    print(f"  Settling time       : {agg['settling_time_s_mean']:.2f} s  (achieved in {agg['settling_time_s_success']*100:.0f}% of episodes)")
    print(f"  Steady-state error  : {agg['steady_state_error_m_mean']:.3f} ± {agg['steady_state_error_m_std']:.3f} m")
    print(f"  Hover time          : {agg['hover_time_s_mean']:.2f} ± {agg['hover_time_s_std']:.2f} s")
    print(f"  Max tilt            : {agg['max_tilt_deg_mean']:.1f} ± {agg['max_tilt_deg_std']:.1f} °")
    print(f"  Control effort      : {agg['ctrl_effort_mean']:.4f}")
    print(f"  Mean speed          : {agg['mean_speed_ms_mean']:.3f} m/s")

    output = {
        "model":     args.model,
        "n_episodes": args.episodes,
        "aggregate": agg,
        "episodes":  episodes,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
