import torch


def compute_gae(
    rewards,
    values,
    dones,
    gamma=0.99,
    lam=0.95,
):
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)

    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

def policy_loss(logprobs_new, logprobs_old , advantages , clip_epsilon=0.2):
    ratios = torch.exp(logprobs_new - logprobs_old)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    loss = -torch.min(surr1, surr2).mean()
    return loss 

def value_loss(values_new, returns):
    return torch.mean((values_new - returns) ** 2)
    
def entropy_bonus(logits):
    return torch.mean(logits)

def total_loss(
    policy_loss,
    value_loss,
    entropy,
    c1=0.5,
    c2=0.01,
):
    return policy_loss + c1 * value_loss - c2 * entropy















