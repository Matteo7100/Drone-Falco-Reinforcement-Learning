import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from quad_env import QuadcopterEnv
import tkinter as tk
from tkinter import ttk

# =============================================================================
#  PPO HYPERPARAMETERS
# =============================================================================
EPOCHS          = 1000       # total training epochs
STEPS_PER_EPOCH = 6000       # environment steps collected before each update
GAMMA           = 0.99       # discount factor (higher = more far-sighted)
LAM             = 0.95       # GAE lambda
CLIP_RATIO      = 0.2        # PPO clipping
PI_LR           = 2e-4       # actor learning rate
VF_LR           = 6e-4       # critic learning rate
TRAIN_ITERS     = 80         # gradient steps per epoch
TARGET_KL       = 0.015      # early stop threshold
ENT_COEF        = 0.01       # entropy bonus (encourages exploration)
MAX_GRAD_NORM   = 0.5        # gradient clipping

SAVE_EVERY      = 50         # save checkpoint every N epochs
MODEL_PATH      = "quad_model.pth"

# =============================================================================
#  NETWORK
# =============================================================================
class ActorCritic(nn.Module):
    """
    Larger network (256 units) than the original (64).
    Shared nothing between actor and critic — cleaner separation.
    log_std is a learnable parameter clamped to reasonable range.
    """
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

        # Initialise with small weights → start near zero mean action (hover)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # Output layer of actor: small gain → initial actions ≈ 0 ≡ hover
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

        # log_std: start at -0.5 → moderate initial exploration
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, x):
        mu    = self.actor(x)
        value = self.critic(x).squeeze(-1)
        std   = torch.exp(torch.clamp(self.log_std, -3.0, 0.5))
        return mu, std, value

    def get_action(self, obs: torch.Tensor):
        mu, std, value = self.forward(obs)
        dist    = Normal(mu, std)
        action  = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mu, std, value = self.forward(obs)
        dist     = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, value


# =============================================================================
#  GAE
# =============================================================================
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """
    Compute Generalised Advantage Estimates.
    Handles episode boundaries correctly (dones mask).
    """
    rewards    = np.array(rewards,  dtype=np.float32)
    values     = np.array(values,   dtype=np.float32)
    dones      = np.array(dones,    dtype=np.float32)
    T          = len(rewards)
    adv        = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0
    next_val   = float(last_value)

    for t in reversed(range(T)):
        mask       = 1.0 - dones[t]
        delta      = rewards[t] + gamma * mask * next_val - values[t]
        last_adv   = delta + gamma * lam * mask * last_adv
        adv[t]     = last_adv
        next_val   = values[t]

    returns = adv + values
    return adv, returns


# =============================================================================
#  PROGRESS UI
# =============================================================================
def make_progress_window(epochs):
    root         = tk.Tk()
    root.title("PPO Training Progress")
    progress_var = tk.DoubleVar()
    bar          = ttk.Progressbar(root, variable=progress_var, maximum=epochs, length=500)
    bar.pack(padx=20, pady=(20, 5))
    label        = tk.Label(root, text="Starting training...", font=("Helvetica", 10))
    label.pack(pady=(0, 20))
    root.update()
    return root, progress_var, label


# =============================================================================
#  TRAINING LOOP
# =============================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Render=False → p.DIRECT (much faster for training)
    env     = QuadcopterEnv(render=False)
    obs_dim = env.observation_space.shape[0]   # 12
    act_dim = env.action_space.shape[0]        # 4

    model        = ActorCritic(obs_dim, act_dim).to(device)
    pi_optimizer = optim.Adam(model.actor.parameters(),  lr=PI_LR, eps=1e-5)
    vf_optimizer = optim.Adam(model.critic.parameters(), lr=VF_LR, eps=1e-5)

    # LR annealing — linearly decay to 10% over training
    pi_scheduler = optim.lr_scheduler.LinearLR(pi_optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS)
    vf_scheduler = optim.lr_scheduler.LinearLR(vf_optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS)

    root, progress_var, status_label = make_progress_window(EPOCHS)

    best_return = -np.inf

    for epoch in range(EPOCHS):
        # ── Collect trajectory ───────────────────────────────────────────────
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        obs      = env.reset()
        ep_ret   = 0.0
        ep_len   = 0
        ep_rets  = []          # track returns for logging

        for t in range(STEPS_PER_EPOCH):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, logp, value = model.get_action(obs_t)

            action_np = action.cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)

            obs_buf.append(obs.copy())
            act_buf.append(action_np)
            logp_buf.append(logp.cpu().item())
            rew_buf.append(reward)
            val_buf.append(value.cpu().item())
            done_buf.append(float(done))

            obs      = next_obs
            ep_ret  += reward
            ep_len  += 1

            if done:
                ep_rets.append(ep_ret)
                ep_ret  = 0.0
                ep_len  = 0
                obs     = env.reset()

        # Bootstrap last value
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, last_val = model.get_action(obs_t)
        last_val = last_val.cpu().item()

        # ── Compute advantages ───────────────────────────────────────────────
        adv_buf, ret_buf = compute_gae(rew_buf, val_buf, done_buf, last_val, GAMMA, LAM)

        # Convert to tensors
        obs_t    = torch.as_tensor(np.array(obs_buf),  dtype=torch.float32, device=device)
        act_t    = torch.as_tensor(np.array(act_buf),  dtype=torch.float32, device=device)
        logp_t   = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_t    = torch.as_tensor(adv_buf,             dtype=torch.float32, device=device)
        ret_t    = torch.as_tensor(ret_buf,             dtype=torch.float32, device=device)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        # ── PPO update ───────────────────────────────────────────────────────
        for ppo_iter in range(TRAIN_ITERS):
            logp_new, entropy, values = model.evaluate_actions(obs_t, act_t)

            ratio    = torch.exp(logp_new - logp_t)
            clip_adv = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * adv_t
            pi_loss  = -(torch.min(ratio * adv_t, clip_adv) + ENT_COEF * entropy).mean()
            vf_loss  = ((ret_t - values) ** 2).mean()

            pi_optimizer.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(model.actor.parameters(), MAX_GRAD_NORM)
            pi_optimizer.step()

            vf_optimizer.zero_grad()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), MAX_GRAD_NORM)
            vf_optimizer.step()

            # KL early stop
            with torch.no_grad():
                approx_kl = (logp_t - logp_new).mean().item()
            if approx_kl > 1.5 * TARGET_KL:
                print(f"  KL early stop at PPO iter {ppo_iter}: KL={approx_kl:.4f}")
                break

        pi_scheduler.step()
        vf_scheduler.step()

        # ── Logging ──────────────────────────────────────────────────────────
        mean_ret  = float(np.mean(ep_rets)) if ep_rets else ep_ret
        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"MeanReturn={mean_ret:8.2f} | "
              f"Episodes={len(ep_rets):3d} | "
              f"KL={approx_kl:.4f} | "
              f"pi_loss={pi_loss.item():.4f} | "
              f"vf_loss={vf_loss.item():.4f}")

        # UI update
        progress_var.set(epoch + 1)
        status_label.config(text=f"Epoch {epoch+1}/{EPOCHS}  |  MeanReturn={mean_ret:.1f}")
        root.update()

        # ── Save checkpoint ──────────────────────────────────────────────────
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Checkpoint saved → {MODEL_PATH}")

        if mean_ret > best_return:
            best_return = mean_ret
            torch.save(model.state_dict(), "quad_model_best.pth")

    # ── Final save ───────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}")
    env.close()

    status_label.config(text=f"Done! Best return: {best_return:.1f}")
    root.update()
    root.after(4000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    train()
