import torch
import time
from RLLM.envs.cartpole import CartPoleEnv
from RLLM.models.policy import ActorCritic
from RLLM.rollout import RolloutBuffer
from RLLM.losses import policy_loss, value_loss, total_loss, compute_gae
from RLLM.chkpt import save_checkpoint , load_checkpoint

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

ROLLOUT_LEN = 2048
NUM_UPDATES = 100
LOG_INTERVAL = 10

CHECKPIONT_INTERVAL = 50
CHECKPIONT_DIR = "checkpoints"
RESUME_PATH = "checkpoints/ckpt_000050.pt"


start_update = 0 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = CartPoleEnv(device=device, seed=42)

model = ActorCritic(obs_dim=env.obs_dim, action_dim=env.action_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

buffer = RolloutBuffer(rollout_len=ROLLOUT_LEN, obs_dim=env.obs_dim, device=device)

obs = env.reset()

if RESUME_PATH is not None:
    print(f"Resuming from checkpoint: {RESUME_PATH}")
    start_update = load_checkpoint(
        RESUME_PATH,
        model,
        optimizer,
        device,
    )

try:
    for update in range(start_update, NUM_UPDATES):
        
        buffer.reset()
        episode_rewards = []
        current_episode_reward = 0.0
        
        rollout_start = time.perf_counter()
        while not buffer.full:
            with torch.no_grad():
                action, logprob, value = model(obs)
    
            next_obs, reward, done, _ = env.step(action)
            current_episode_reward += reward.item()
    
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                logprob=logprob.detach(),
                value=value.detach(),
            )
    
            obs = next_obs
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs = env.reset()
        
        rollout_end = time.perf_counter()
        rollout_time = rollout_end - rollout_start
        steps_per_sec = ROLLOUT_LEN / rollout_time
    
        data = buffer.get()
    
        advantages, returns = compute_gae(
            data["rewards"],
            data["values"],
            data["dones"],
        )
    
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    
        action_new, logprob_new, value_new = model(data["obs"])
        entropy = model.get_entropy(data["obs"])
    
        p_loss = policy_loss(
            logprob_new,
            data["logprobs"],
            advantages,
        )
        
        v_loss = value_loss(value_new, returns)
        ent = entropy.mean()
        
        loss = total_loss(p_loss, v_loss, ent)
    
        if not torch.isfinite(loss):
            print(f"ERROR: Non-finite loss detected at update {update}")
            break
    
        optimizer.zero_grad()
        
        update_start = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        update_end = time.perf_counter()
        update_time_ms = (update_end - update_start) * 1000
    
        if update % LOG_INTERVAL == 0:
            mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            
            print(f"Update {update:4d} | "
                  f"Reward: {mean_reward:7.2f} | "
                  f"Rollout: {ROLLOUT_LEN} steps in {rollout_time:.3f}s ({steps_per_sec:.1f} steps/sec) | "
                  f"Update: {update_time_ms:.1f} ms | "
                  f"Entropy: {ent.item():6.4f}")
            
        if update % CHECKPIONT_INTERVAL == 0 and update > 0:
            ckpt_path = f"{CHECKPIONT_DIR}/ckpt_{update:06d}.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                update,
            )
    
            try:
                buffer.check_finite()
            except RuntimeError as e:
                print(f"ERROR: {e}")
                break

except KeyboardInterrupt:
    print("Interrupted. Saving checkpoint...")
    ckpt_path = f"{CHECKPIONT_INTERVAL}/ckpt_interrupt.pt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        update,
    )
    raise

