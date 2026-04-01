from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from prepare import DATASET_DIR, ENV_ID, TIME_BUDGET


@dataclass
class TrainConfig:
    data_dir: str = DATASET_DIR
    env_id: str = ENV_ID
    time_budget_seconds: int = TIME_BUDGET
    total_eval_episodes: int = 10
    num_envs: int = 8
    rollout_steps: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2.7e-4
    lr_final_frac: float = 0.5
    anneal_lr: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    entropy_coef: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 6
    minibatches: int = 4
    target_kl: float | None = 0.02
    normalize_advantages: bool = True
    hidden_size: int = 256
    hidden_layers: int = 2
    activation: str = "tanh"
    ortho_init: bool = True
    obs_norm: bool = True
    obs_norm_clip: float = 10.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-5
    seed: int = 1337
    device: str = "cuda"
    compile: bool = False
    log_interval: int = 10
    eval_interval_seconds: float = 30.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError("activation must be one of: relu, tanh, silu")


def layer_init(layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: TrainConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for _ in range(config.hidden_layers):
            linear = nn.Linear(input_dim, config.hidden_size)
            if config.ortho_init:
                linear = layer_init(linear)
            layers.extend([linear, make_activation(config.activation)])
            input_dim = config.hidden_size

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, action_dim)
        self.value_head = nn.Linear(input_dim, 1)
        if config.ortho_init:
            layer_init(self.policy_head, std=0.01)
            layer_init(self.value_head, std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(obs)
        return value


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )


def load_eval_seeds(data_dir: Path) -> list[int]:
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        seeds = metadata.get("eval_seeds")
        if isinstance(seeds, list) and seeds:
            return [int(seed) for seed in seeds]
    return [101, 203, 307, 401, 509]


def choose_num_minibatches(batch_size: int, requested_minibatches: int) -> int:
    upper = min(batch_size, requested_minibatches)
    for candidate in range(upper, 0, -1):
        if batch_size % candidate == 0:
            return candidate
    return 1


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count


def normalize_obs(
    obs: np.ndarray,
    obs_rms: RunningMeanStd | None,
    clip_value: float,
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs_rms is None:
        return obs
    normalized = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
    return np.clip(normalized, -clip_value, clip_value).astype(np.float32)


def collect_completed_episodes(
    rewards: np.ndarray,
    dones: np.ndarray,
    running_returns: np.ndarray,
    running_lengths: np.ndarray,
    completed_returns: list[float],
    completed_lengths: list[int],
) -> None:
    running_returns += rewards.astype(np.float32)
    running_lengths += 1
    done_indices = np.nonzero(dones)[0]
    for index in done_indices:
        completed_returns.append(float(running_returns[index]))
        completed_lengths.append(int(running_lengths[index]))
        running_returns[index] = 0.0
        running_lengths[index] = 0


def make_env(env_id: str, seed: int | None = None) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    env_id: str,
    device: torch.device,
    total_eval_episodes: int,
    eval_seeds: list[int],
    obs_rms: RunningMeanStd | None,
    obs_norm_clip: float,
) -> dict[str, float]:
    model.eval()
    returns: list[float] = []
    lengths: list[int] = []

    for episode_idx in range(total_eval_episodes):
        seed = eval_seeds[episode_idx % len(eval_seeds)] + episode_idx
        env = make_env(env_id, seed=seed)
        obs, _ = env.reset(seed=seed)
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (done or truncated):
            obs_tensor = torch.as_tensor(
                normalize_obs(obs, obs_rms, obs_norm_clip),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            logits, _ = model(obs_tensor)
            action = torch.argmax(logits, dim=-1).item()
            obs, reward, done, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)
        env.close()

    returns_array = np.asarray(returns, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.float32)
    success_rate = float(np.mean(returns_array >= 200.0))
    return {
        "eval_return_mean": float(returns_array.mean()),
        "eval_return_std": float(returns_array.std()),
        "eval_success_rate": success_rate,
        "eval_episode_length_mean": float(lengths_array.mean()),
    }


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if torch.isclose(var_y, torch.tensor(0.0, device=y_true.device)):
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LunarLander-v3 + PPO")
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--env-id", type=str, default=TrainConfig.env_id)
    parser.add_argument("--time-budget-seconds", type=int, default=TrainConfig.time_budget_seconds)
    parser.add_argument("--total-eval-episodes", type=int, default=TrainConfig.total_eval_episodes)
    parser.add_argument("--num-envs", type=int, default=TrainConfig.num_envs)
    parser.add_argument("--rollout-steps", type=int, default=TrainConfig.rollout_steps)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=TrainConfig.gae_lambda)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--lr-final-frac", type=float, default=TrainConfig.lr_final_frac)
    parser.add_argument("--clip-coef", type=float, default=TrainConfig.clip_coef)
    parser.add_argument("--entropy-coef", type=float, default=TrainConfig.entropy_coef)
    parser.add_argument("--value-coef", type=float, default=TrainConfig.value_coef)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--update-epochs", type=int, default=TrainConfig.update_epochs)
    parser.add_argument("--minibatches", type=int, default=TrainConfig.minibatches)
    parser.add_argument("--target-kl", type=float, default=TrainConfig.target_kl)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--hidden-layers", type=int, default=TrainConfig.hidden_layers)
    parser.add_argument("--activation", type=str, default=TrainConfig.activation)
    parser.add_argument("--obs-norm-clip", type=float, default=TrainConfig.obs_norm_clip)
    parser.add_argument("--adam-beta1", type=float, default=TrainConfig.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=TrainConfig.adam_beta2)
    parser.add_argument("--adam-eps", type=float, default=TrainConfig.adam_eps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--eval-interval-seconds", type=float, default=TrainConfig.eval_interval_seconds)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages", action="store_false")
    parser.add_argument("--ortho-init", action="store_true")
    parser.add_argument("--no-ortho-init", dest="ortho_init", action="store_false")
    parser.add_argument("--obs-norm", action="store_true")
    parser.add_argument("--no-obs-norm", dest="obs_norm", action="store_false")
    parser.set_defaults(
        anneal_lr=TrainConfig.anneal_lr,
        clip_vloss=TrainConfig.clip_vloss,
        normalize_advantages=TrainConfig.normalize_advantages,
        ortho_init=TrainConfig.ortho_init,
        obs_norm=TrainConfig.obs_norm,
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = TrainConfig(**vars(args))
    if config.minibatches < 1:
        raise ValueError("minibatches must be >= 1")
    if config.num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if config.rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    set_seed(config.seed)

    data_root = Path(config.data_dir)
    if not data_root.exists():
        raise FileNotFoundError(
            f"experiment metadata not prepared under {data_root}. Run `python prepare.py` first."
        )
    eval_seeds = load_eval_seeds(data_root)

    def thunk(index: int):
        def _factory() -> gym.Env:
            return make_env(config.env_id, seed=config.seed + index)

        return _factory

    envs = gym.vector.SyncVectorEnv([thunk(i) for i in range(config.num_envs)])
    single_action_space = envs.single_action_space
    single_observation_space = envs.single_observation_space
    if not isinstance(single_action_space, gym.spaces.Discrete):
        raise TypeError("this baseline only supports discrete action spaces")
    if not isinstance(single_observation_space, gym.spaces.Box):
        raise TypeError("this baseline expects a Box observation space")

    obs_dim = int(np.prod(single_observation_space.shape))
    action_dim = int(single_action_space.n)
    batch_size = config.num_envs * config.rollout_steps
    if batch_size % config.minibatches != 0:
        raise ValueError("num_envs * rollout_steps must be divisible by minibatches")
    minibatch_size = batch_size // config.minibatches

    model = ActorCritic(obs_dim, action_dim, config).to(device)
    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = build_optimizer(config, model)
    obs_rms = RunningMeanStd((obs_dim,)) if config.obs_norm else None

    obs = torch.zeros((config.rollout_steps, config.num_envs, obs_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)

    next_obs_np, _ = envs.reset(seed=config.seed)
    if obs_rms is not None:
        obs_rms.update(next_obs_np)
    next_obs = torch.as_tensor(
        normalize_obs(next_obs_np, obs_rms, config.obs_norm_clip),
        dtype=torch.float32,
        device=device,
    )
    next_done = torch.zeros(config.num_envs, dtype=torch.float32, device=device)

    print(
        f"env={config.env_id} obs_dim={obs_dim} actions={action_dim} dev={device} "
        f"num_envs={config.num_envs} rollout={config.rollout_steps} batch={batch_size} "
        f"minibatch={minibatch_size} obs_norm={config.obs_norm} budget={config.time_budget_seconds}s"
    )

    training_start = time.time()
    last_eval_time = training_start
    global_step = 0
    update = 0
    episode_returns_window: list[float] = []
    episode_lengths_window: list[int] = []
    best_eval_return = float("-inf")
    best_success_rate = 0.0
    running_episode_returns = np.zeros(config.num_envs, dtype=np.float32)
    running_episode_lengths = np.zeros(config.num_envs, dtype=np.int32)

    while time.time() - training_start < config.time_budget_seconds:
        update += 1
        if config.anneal_lr:
            progress = min((time.time() - training_start) / config.time_budget_seconds, 1.0)
            lr_now = config.lr * (1.0 - progress * (1.0 - config.lr_final_frac))
            optimizer.param_groups[0]["lr"] = lr_now
        else:
            lr_now = optimizer.param_groups[0]["lr"]

        steps_collected = 0
        for step in range(config.rollout_steps):
            elapsed = time.time() - training_start
            if elapsed >= config.time_budget_seconds:
                break

            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(
                action.cpu().numpy()
            )
            done_np = np.logical_or(terminated_np, truncated_np)
            collect_completed_episodes(
                rewards=reward_np,
                dones=done_np,
                running_returns=running_episode_returns,
                running_lengths=running_episode_lengths,
                completed_returns=episode_returns_window,
                completed_lengths=episode_lengths_window,
            )
            rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            if obs_rms is not None:
                obs_rms.update(next_obs_np)
            next_obs = torch.as_tensor(
                normalize_obs(next_obs_np, obs_rms, config.obs_norm_clip),
                dtype=torch.float32,
                device=device,
            )
            next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)
            episode_returns_window = episode_returns_window[-100:]
            episode_lengths_window = episode_lengths_window[-100:]
            steps_collected += 1

        if steps_collected == 0:
            break

        with torch.no_grad():
            next_value = model.get_value(next_obs)
            rewards_slice = rewards[:steps_collected]
            dones_slice = dones[:steps_collected]
            values_slice = values[:steps_collected]
            advantages = torch.zeros_like(rewards_slice)
            last_gae = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
            for step in reversed(range(steps_collected)):
                if step == steps_collected - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones_slice[step + 1]
                    next_values = values_slice[step + 1]
                delta = (
                    rewards_slice[step]
                    + config.gamma * next_values * next_non_terminal
                    - values_slice[step]
                )
                last_gae = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae
                advantages[step] = last_gae
            returns = advantages + values_slice

        b_obs = obs[:steps_collected].reshape((-1, obs_dim))
        b_actions = actions[:steps_collected].reshape(-1)
        b_logprobs = logprobs[:steps_collected].reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_slice.reshape(-1)

        current_batch_size = steps_collected * config.num_envs
        current_minibatches = choose_num_minibatches(current_batch_size, config.minibatches)
        current_minibatch_size = current_batch_size // current_minibatches
        batch_indices = np.arange(current_batch_size)
        clipfracs: list[float] = []
        approx_kl = 0.0
        entropy_loss_value = 0.0
        pg_loss_value = 0.0
        value_loss_value = 0.0

        model.train()
        for epoch in range(config.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, current_batch_size, current_minibatch_size):
                end = start + current_minibatch_size
                mb_idx = batch_indices[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )
                logratio = newlogprob - b_logprobs[mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - logratio).mean().item())
                    clipfracs.append(
                        float(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                    )

                mb_advantages = b_advantages[mb_idx]
                if config.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - config.clip_coef,
                    1.0 + config.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_idx]).pow(2)
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        newvalue - b_values[mb_idx],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (newvalue - b_returns[mb_idx]).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.entropy_coef * entropy_loss + config.value_coef * value_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                entropy_loss_value = float(entropy_loss.item())
                pg_loss_value = float(pg_loss.item())
                value_loss_value = float(value_loss.item())

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        model.eval()
        y_pred = b_values.detach()
        y_true = b_returns.detach()
        ev = explained_variance(y_pred, y_true)
        train_return_mean = (
            float(np.mean(episode_returns_window)) if episode_returns_window else float("nan")
        )
        train_episode_length_mean = (
            float(np.mean(episode_lengths_window)) if episode_lengths_window else float("nan")
        )
        now = time.time()

        if update % config.log_interval == 0:
            print(
                f"[up{update:04d}] "
                f"step={global_step:07d} "
                f"lr={lr_now:.6f} "
                f"train_return_mean={train_return_mean:.2f} "
                f"train_ep_len_mean={train_episode_length_mean:.1f} "
                f"pg_loss={pg_loss_value:.4f} "
                f"v_loss={value_loss_value:.4f} "
                f"entropy={entropy_loss_value:.4f} "
                f"approx_kl={approx_kl:.4f} "
                f"clipfrac={float(np.mean(clipfracs)) if clipfracs else 0.0:.4f} "
                f"explained_var={ev:.4f} "
                f"time={now - training_start:.1f}/{config.time_budget_seconds}s"
            )

        if now - last_eval_time >= config.eval_interval_seconds:
            eval_metrics = evaluate_policy(
                model=model,
                env_id=config.env_id,
                device=device,
                total_eval_episodes=config.total_eval_episodes,
                eval_seeds=eval_seeds,
                obs_rms=obs_rms,
                obs_norm_clip=config.obs_norm_clip,
            )
            best_eval_return = max(best_eval_return, eval_metrics["eval_return_mean"])
            best_success_rate = max(best_success_rate, eval_metrics["eval_success_rate"])
            print(
                f"[eval] up={update:04d} "
                f"return_mean={eval_metrics['eval_return_mean']:.2f} "
                f"return_std={eval_metrics['eval_return_std']:.2f} "
                f"success_rate={eval_metrics['eval_success_rate']:.2f} "
                f"ep_len={eval_metrics['eval_episode_length_mean']:.1f}"
            )
            last_eval_time = now

    final_metrics = evaluate_policy(
        model=model,
        env_id=config.env_id,
        device=device,
        total_eval_episodes=config.total_eval_episodes,
        eval_seeds=eval_seeds,
        obs_rms=obs_rms,
        obs_norm_clip=config.obs_norm_clip,
    )
    best_eval_return = max(best_eval_return, final_metrics["eval_return_mean"])
    best_success_rate = max(best_success_rate, final_metrics["eval_success_rate"])

    print(f"best eval return mean: {best_eval_return:.2f}")
    print(f"best eval success rate: {best_success_rate:.2f}")
    print(f"final eval return mean: {final_metrics['eval_return_mean']:.2f}")
    print(f"final eval return std: {final_metrics['eval_return_std']:.2f}")
    print(f"final eval success rate: {final_metrics['eval_success_rate']:.2f}")
    print(f"final eval episode length mean: {final_metrics['eval_episode_length_mean']:.1f}")
    print(f"total updates: {update}")
    print(f"total env steps: {global_step}")
    print(f"training time: {time.time() - training_start:.1f}s")

    envs.close()


if __name__ == "__main__":
    main()
