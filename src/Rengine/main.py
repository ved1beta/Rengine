import torch
import time
import numpy as np
import random
from Rengine.envs.cartpole import CartPoleEnv
from Rengine.models.policy import ActorCritic
from Rengine.rollout import RolloutBuffer
from Rengine.losses import policy_loss, value_loss, total_loss, compute_gae
from Rengine.chkpt import save_checkpoint , load_checkpoint

def train(rank=0, world_size=1):
    base_seed = 42
    seed = base_seed + rank

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    is_master = (rank == 0)

    if is_master:
        print(f"Using device: {device}")

    ROLLOUT_LEN = 2048
    NUM_UPDATES = 500
    LOG_INTERVAL = 10

    CHECKPIONT_INTERVAL = 50
    CHECKPIONT_DIR = "checkpoints"
    RESUME_PATH = None

    start_update = 0 

    env = CartPoleEnv(device=device, seed=seed)

    model = ActorCritic(obs_dim=env.obs_dim, action_dim=env.action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    buffer = RolloutBuffer(rollout_len=ROLLOUT_LEN, obs_dim=env.obs_dim, device=device)

    obs = env.reset()

    if RESUME_PATH is not None:
        if is_master:
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
                if is_master:
                    print(f"ERROR: Non-finite loss detected at update {update}")
                break
        
            optimizer.zero_grad()
            
            update_start = time.perf_counter()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            update_end = time.perf_counter()
            update_time_ms = (update_end - update_start) * 1000
        
            if update % LOG_INTERVAL == 0 and is_master:
                mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
                
                print(f"Update {update:4d} | "
                      f"Reward: {mean_reward:7.2f} | "
                      f"Rollout: {ROLLOUT_LEN} steps in {rollout_time:.3f}s ({steps_per_sec:.1f} steps/sec) | "
                      f"Update: {update_time_ms:.1f} ms | "
                      f"Entropy: {ent.item():6.4f}")
                
            if update % CHECKPIONT_INTERVAL == 0 and update > 0 and is_master:
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
        if is_master:
            print("Interrupted. Saving checkpoint...")
            ckpt_path = f"{CHECKPIONT_INTERVAL}/ckpt_interrupt.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                update,
            )
        raise

if __name__ == "__main__":
    train()

