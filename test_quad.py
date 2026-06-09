import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from quad_env import QuadcopterEnv
import time
import argparse

# =============================================================================
#  MUST MATCH the architecture used during training exactly
# =============================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 256), nn.Tanh(),
                nn.Linear(256, 256),   nn.Tanh(),
                nn.Linear(256, out_dim)
            )

        self.actor  = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, x):
        mu    = self.actor(x)
        value = self.critic(x).squeeze(-1)
        std   = torch.exp(torch.clamp(self.log_std, -3.0, 0.5))
        return mu, std, value

    def get_action_deterministic(self, obs: torch.Tensor):
        """At test time use the mean action — no sampling noise."""
        mu, _, value = self.forward(obs)
        return mu, value


# =============================================================================
#  METRICS TRACKER
# =============================================================================
class EpisodeMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reward   = 0.0
        self.steps          = 0
        self.distances      = []
        self.min_dist       = np.inf
        self.termination    = "timeout"

    def update(self, reward, dist, done, info):
        self.total_reward += reward
        self.steps        += 1
        self.distances.append(dist)
        self.min_dist      = min(self.min_dist, dist)
        if done and 'termination' in info:
            self.termination = info['termination']

    def summary(self, ep_num):
        avg_dist = float(np.mean(self.distances)) if self.distances else 0.0
        print(f"\n{'─'*50}")
        print(f"  Episode {ep_num} summary")
        print(f"  Total reward   : {self.total_reward:8.2f}")
        print(f"  Steps          : {self.steps}")
        print(f"  Avg distance   : {avg_dist:.3f} m")
        print(f"  Min distance   : {self.min_dist:.3f} m")
        print(f"  Termination    : {self.termination}")
        print(f"{'─'*50}")


# =============================================================================
#  TEST LOOP
# =============================================================================
def test(model_path: str, num_episodes: int, slow_down: float, stochastic: bool):
    device = torch.device("cpu")   # CPU is fine for inference

    # Load environment with GUI rendering
    env     = QuadcopterEnv(render=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load model
    model = ActorCritic(obs_dim, act_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from '{model_path}'")
    print(f"Target position : {env.target_position}")
    print(f"Stochastic mode : {stochastic}")
    print(f"Episodes        : {num_episodes}\n")

    metrics = EpisodeMetrics()

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        metrics.reset()
        done = False

        print(f"Episode {ep} — running...")

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                if stochastic:
                    # Sample from the policy distribution
                    mu, std, value = model(obs_t)
                    from torch.distributions import Normal
                    action = Normal(mu, std).sample()
                else:
                    # Pure greedy — mean action
                    action, value = model.get_action_deterministic(obs_t)

            action_np = action.numpy()
            obs, reward, done, info = env.step(action_np)

            dist = info.get('dist', 0.0)
            metrics.update(reward, dist, done, info)

            # Optional slow-down so you can watch in the GUI
            if slow_down > 0:
                time.sleep(slow_down)

        metrics.summary(ep)

    env.close()
    print("\nAll episodes complete.")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained quadcopter PPO model")
    parser.add_argument(
        "--model",
        type=str,
        default="quad_model_best.pth",
        help="Path to the saved model weights (default: quad_model_best.pth)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes to run (default: 5)"
    )
    parser.add_argument(
        "--slow",
        type=float,
        default=0.01,
        help="Seconds to sleep between steps — use 0 for max speed (default: 0.01)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy distribution instead of taking the mean action"
    )
    args = parser.parse_args()

    test(
        model_path   = args.model,
        num_episodes = args.episodes,
        slow_down    = args.slow,
        stochastic   = args.stochastic,
    )
