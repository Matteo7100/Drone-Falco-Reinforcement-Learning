import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from quad_env import QuadcopterEnv  # Ensure this is your custom env file
import tkinter as tk
from tkinter import ttk



# PPO Hyperparameters
epochs = 1000
steps_per_epoch = 6000
gamma = 0.95
lam = 0.95
clip_ratio = 0.2
pi_lr = 3e-4   # A more standard PPO learning rate
vf_lr = 1e-3   # Critic can often learn faster
train_iters = 80 # More update steps per batch of data
target_kl = 0.015

# Create popup window
root = tk.Tk()
root.title("PPO Training Progress")
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=epochs, length=400)
progress_bar.pack(padx=20, pady=20)
status_label = tk.Label(root, text="Starting training...")
status_label.pack()
root.update()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)  

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        # compute mean and value
        value = self.critic(x)  # Critic output
        mu = self.actor(x)      # Actor output
        std = torch.exp(self.log_std)   # Standard deviation exponentiated
        return mu, std, value

    def get_action(self, obs):
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_probs, entropy, value

def compute_gae(rewards, values, dones, next_value):
    adv = np.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - dones[t]) * next_value - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        adv[t] = last_adv
        next_value = values[t]
    return adv

# Initialize environment and model
env = QuadcopterEnv()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
model = ActorCritic(obs_dim, act_dim).to(device)
pi_optimizer = optim.Adam(model.actor.parameters(), lr=pi_lr)
vf_optimizer = optim.Adam(model.critic.parameters(), lr=vf_lr)

# PPO Training Loop
for epoch in range(epochs):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()     # Reset the environment
    
    ep_ret, ep_len = 0, 0


    for t in range(steps_per_epoch):
        obs = np.asarray(obs, dtype=np.float32)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device) 

        #update progress bar
        progress_var.set(epoch + 1)
        status_label.config(text=f"Epoch {epoch + 1} of {epochs}")
        root.update()

        with torch.no_grad():
            action, logp, value = model.get_action(obs_tensor)
        action_np = action.cpu().numpy()

        next_obs, reward, done, rewards = env.step(action_np)

        obs_buf.append(obs)
        act_buf.append(action_np)
        logp_buf.append(logp.cpu().numpy())
        rew_buf.append(reward)
        val_buf.append(value.cpu().numpy())
        done_buf.append(done)

        obs = next_obs
        ep_ret += reward
        ep_len += 1

        if done or (t == steps_per_epoch - 1):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                next_value = model(obs_tensor)[2].cpu().numpy()


            adv_buf = compute_gae(rew_buf, val_buf, done_buf, next_value)            
            
            ret_buf = adv_buf + np.array(val_buf)

            get_state = lambda: np.array(env._get_state(), dtype=np.float32)
            

            obs_buf = np.array(obs_buf, dtype=np.float32)
            act_buf = np.array(act_buf, dtype=np.float32)
            logp_buf = np.array(logp_buf, dtype=np.float32)
            rew_buf = np.array(ret_buf, dtype=np.float32)
            val_buf = np.array(adv_buf, dtype=np.float32)


            obs_t = torch.as_tensor(obs_buf, dtype=torch.float32).to(device)
            act_t = torch.as_tensor(act_buf, dtype=torch.float32).to(device)
            adv_t = torch.as_tensor(adv_buf, dtype=torch.float32).to(device)
            ret_t = torch.as_tensor(ret_buf, dtype=torch.float32).to(device)
            logp_old_t = torch.as_tensor(logp_buf, dtype=torch.float32).to(device)

            # normalizes the tensor of advantages -> subtracts the mean and divides by the standard deviation 
            # (with a small constant added for numerical stability). stabilize and speed up training.
            # This is a common practice in PPO to ensure that the advantage estimates are centered around zero and have unit variance.
            # Disabling Bessel’s correction(unbiased = False), using N instead of N - 1 for the denominator allows to not have denominator 
            # equal to zero or negative, which is mathematically invalid for standard deviation
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

            for _ in range(train_iters):
                logp, entropy, value = model.evaluate_actions(obs_t, act_t)
                ratio = torch.exp(logp - logp_old_t)
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_t
                policy_loss = -(torch.min(ratio * adv_t, clip_adv)).mean()
                value_loss = ((ret_t - value) ** 2).mean()

                pi_optimizer.zero_grad()
                policy_loss.backward()
                pi_optimizer.step()

                vf_optimizer.zero_grad()
                value_loss.backward()
                vf_optimizer.step()
                # "approx_kl" stands for the approximate Kullback-Leibler (KL) divergence. It's a measure of how much the new policy distribution has diverged
                # from the old one. The code compares the mean log probabilities of actions under the old and new policies, and if the divergence exceeds a 
                # threshold (1.5 times target_kl), the training loop stops early.
                approx_kl = (logp_old_t - logp).mean().item()
                if approx_kl > 1.5 * target_kl:
                    print(f"Early stopping at iter due to KL: {approx_kl:.4f}")
                    break
            
            print(f"Reward: {rewards}")
            print(f"Epoch {epoch + 1}: Return={ep_ret:.2f}, Length={ep_len}")
            obs = env.reset()
            ep_ret, ep_len = 0, 0
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []


status_label.config(text="Training complete! Saving model...")
root.update()

torch.save(model.state_dict(), "quad_model.pth")
print(f"Model saved!")

# Close environment
env.close()

status_label.config(text="Training complete! Model saved.")
root.update()
root.after(3000, root.destroy)  # Auto-close after 3 seconds
root.mainloop()

print("Training finished successfully!")