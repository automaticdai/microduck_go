"""Shared categorical tactical actor and centralized, agent-conditioned PPO critic."""
import torch
from torch import nn
from torch.distributions import Categorical


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,outputs))


class TacticalMAPPO(nn.Module):
    def __init__(self, actor_dim, critic_dim, num_actions=5):
        super().__init__()
        self.actor = mlp(actor_dim,num_actions)
        self.critic = mlp(critic_dim,1)
        # Start mostly with the existing ball tracker, while exploring support skills.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():
            self.actor[-1].bias[0] = 4.

    def distribution(self, obs):
        return Categorical(logits=self.actor(obs))

    def value(self, state):
        return self.critic(state).squeeze(-1)


def generalized_advantages(rewards, values, dones, discounts, last_value, lam=.95, valid=None):
    """All tensors retain [time, arena, agent]; terminal rounds never bootstrap."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        alive = (~dones[t]).to(rewards.dtype).unsqueeze(-1)
        gamma = discounts[t].unsqueeze(-1)
        following = last_value if t == len(rewards)-1 else values[t+1]
        delta = rewards[t]+gamma*alive*following-values[t]
        carry = delta+gamma*lam*alive*carry
        if valid is not None:
            # Exclude a corrupt transition. Earlier valid steps bootstrap from its
            # finite pre-step value, with no GAE trace crossing into the reset episode.
            carry = torch.where(valid[t].unsqueeze(-1),carry,0.)
        advantages[t] = carry
    return advantages,advantages+values


def update(model, optimizer, batch, epochs=4, minibatch_size=2048, entropy_coef=.01, valid=None):
    """PPO ratios and advantages are per agent; critic inputs include the whole match."""
    obs,state,actions,old_logp,old_value,advantages,returns = [x.flatten(0,2) for x in batch]
    if valid is not None:
        keep = valid.unsqueeze(-1).expand(batch[2].shape).flatten()
        obs,state,actions,old_logp,old_value,advantages,returns = [
            x[keep] for x in (obs,state,actions,old_logp,old_value,advantages,returns)]
    if not len(actions):
        raise RuntimeError('No valid MAPPO transitions in rollout')
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)
    metrics=[]
    for _ in range(epochs):
        for ids in torch.randperm(len(actions),device=actions.device).split(minibatch_size):
            dist = model.distribution(obs[ids])
            log_ratio = dist.log_prob(actions[ids])-old_logp[ids]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio*advantages[ids],ratio.clamp(.8,1.2)*advantages[ids]).mean()
            value = model.value(state[ids])
            clipped = old_value[ids]+(value-old_value[ids]).clamp(-.2,.2)
            value_loss = .5*torch.maximum((value-returns[ids]).square(),(clipped-returns[ids]).square()).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss+value_loss-entropy_coef*entropy
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite MAPPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
            optimizer.step()
            kl = ((ratio-1)-log_ratio).mean().detach()
            metrics.append(torch.stack((policy_loss.detach(),value_loss.detach(),entropy.detach(),kl)))
            if kl > .03:
                break
        else:
            continue
        break
    return dict(zip(('policy_loss','value_loss','entropy','kl'),torch.stack(metrics).mean(0).cpu().tolist()))
