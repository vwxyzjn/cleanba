import os
import sys
import time
import queue
import random
import warnings
import threading
from dataclasses import dataclass
from typing import NamedTuple
from collections import deque

import envpool
import psutil
import numpy as np

import jax
import rlax
import flax
import optax
import distrax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    "the name of this experiment"
    seed: int = 1
    "the seed of the experiment"
    track: bool = False
    "if toggled, this experiment will be tracked with Weights and Biases"
    wandb_project_name: str = "ApeX DQN"
    "the wandb's project name"
    wandb_entity: str = None
    "the entity (team) of wandb's project"
    capture_video: bool = False
    "whether to capture videos of the agent performances (check out `videos` folder)"
    save_model: bool = False
    "whether to save model into the `runs/{run_name}` folder"
    upload_model: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    env_id: str = "Breakout-v5"
    "the id of the environment"
    total_timesteps: int = 1000000000
    "total number of atari timesteps of the experiments"
    learning_rate: float = 1e-4
    "the learning rate of the optimizer"
    n_envs: int = 200
    "the number of parallel game environments"
    buffer_size: int = 10000
    "the replay memory buffer size"
    rollout_length: int = 100
    "the number of transitions to collect before sending to the replay memory"
    bootstrap_length: int = 3
    "the number of steps to bootstrap"
    alpha: float = 0.6
    "the alpha value for prioritized experience replay"
    beta: float = 0.4
    "the beta value for prioritized experience replay"
    gamma: float = 0.99
    "the discount factor gamma"
    target_network_frequency: int = 2500
    "the timesteps it takes to update the target network"
    batch_size: int = 512
    "the batch size of sample from the reply memory"
    epsilon: float = 0.4
    "the epsilon value for epsilon greedy exploration"
    epsilon_greedy_alpha: float = 7.0
    "the alpha value for epsilon greedy exploration"
    learning_starts: int = 50_000 
    "the timestep it takes to start learning"
    n_prefetch: int = 16
    "the number of batches to prefetch for learner"
    log_interval: int = 100
    "the log interval"

    actor_device_ids = [0]
    "actor devices"
    learner_device_ids = [0]
    "learner devices"
    num_actor_threads: int = 2


ATARI_MAX_FRAMES = 108_000 // 4

def make_env(env_id, seed, num_envs=1):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=True,  # Espeholt et al., 2018, Tab. G.1
            repeat_action_probability=0,  # Hessel et al., 2022 (Muesli) Tab. 10
            noop_max=30,  # Espeholt et al., 2018, Tab. C.1 "Up to 30 no-ops at the beginning of each episode."
            full_action_space=False,  # Espeholt et al., 2018, Appendix G., "Following related work, experts use game-specific action sets."
            max_episode_steps=ATARI_MAX_FRAMES,  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs
    return thunk


class Batch(NamedTuple):
    obs: np.ndarray
    bootstrap_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    firststeps: np.ndarray
    weights: np.ndarray


class ReplayBuffer:
    """ Prioritized Experience Replay buffer for storing rollouts of trajectories """
    def __init__(
        self,
        capacity: int,
        length: int,
        bootstrap_n: int,
        n_envs: int,
        seed: int,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self.capacity = capacity
        self.length = length
        self.n = bootstrap_n
        self.n_envs = n_envs
        self.pos = 0
        self.alpha = alpha
        self.beta = beta
        self.is_setup = False
        self.full = False
        self.timesteps_seen = 0
        self.rng = np.random.default_rng(seed)

        assert self.capacity % self.n_envs == 0, "capacity must be evenly divisible by n_envs"

    def add(self, priorities: np.ndarray, rollout: NamedTuple):
        assert priorities.shape == (self.n_envs, self.length), "Priorities must be shape (n_envs, length)"
        if not self.is_setup:
            self._setup(rollout)

        indexer = slice(self.pos, self.pos + self.n_envs)
        self.priorities[indexer] = np.array(priorities).copy()
        self.obs[indexer] = np.array(rollout.obs).copy()
        self.actions[indexer] = np.array(rollout.actions).copy()
        self.rewards[indexer] = np.array(rollout.rewards).copy()
        self.actions[indexer] = np.array(rollout.actions).copy()
        self.dones[indexer] = np.array(rollout.dones).copy()
        self.firststeps[indexer] = np.array(rollout.firststeps).copy()

        self.timesteps_seen += self.n_envs * self.length
        self.pos = self.pos + self.n_envs
        if self.pos >= self.capacity:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        assert self.is_setup, "buffer must be setup before sampling"
        inds, weights = self._priority_sampling(batch_size)
        return self._get_samples(inds, weights), inds

    def update_priorities(self, inds: np.ndarray, updated_priorities: np.ndarray):
        r, t = np.unravel_index(inds, self.priorities.shape)
        self.priorities[r, t] = updated_priorities

    def size(self):
        return self.pos if not self.full else self.capacity
    
    def ready(self, min_size: int):
        return self.timesteps_seen > min_size

    def _setup(self, example: NamedTuple):
        assert len(example.rewards.shape) == 2, "Rewards must be scalars (i.e. shape (n_envs, length,)))"

        self._check_memory(example)
        shapes = {k: v.shape[2:] for k, v in example._asdict().items()}
        dtype = {k: v.dtype for k, v in example._asdict().items()}

        self.priorities = np.zeros((self.capacity, self.length), dtype=np.float32)
        self.obs = np.zeros((self.capacity, self.length + self.n, *shapes['obs']), dtype=dtype['obs'])
        self.actions = np.zeros((self.capacity, self.length + self.n, *shapes['actions']), dtype=dtype['actions'])
        self.rewards = np.zeros((self.capacity, self.length + self.n, *shapes['rewards']), dtype=dtype['rewards'])
        self.dones = np.zeros((self.capacity, self.length + self.n, *shapes['dones']), dtype=dtype['dones'])
        self.firststeps = np.zeros((self.capacity, self.length + self.n, *shapes['firststeps']), dtype=dtype['firststeps'])
        self.is_setup = True

    def _priority_sampling(self, batch_size: int):
        N = self.capacity * self.length
        sample_probs = self.priorities.flatten() ** self.alpha
        sample_probs /= sample_probs.sum() 
        sampled_inds = self.rng.choice(N, batch_size, p=sample_probs, replace=False)
        importance_sampling_weights = (N * sample_probs[sampled_inds]) ** -self.beta
        importance_sampling_weights /= importance_sampling_weights.max()
        return sampled_inds, importance_sampling_weights

    def _get_samples(self, inds, weights: np.ndarray):
        r, t = np.unravel_index(inds, self.priorities.shape)
        t_to_n = t[:, None] + np.arange(self.n)[None,] 
        return Batch(
            obs=self.obs[r, t],
            bootstrap_obs=self.obs[r, t + self.n],
            actions=self.actions[r, t],
            rewards=self.rewards[r[:, None], t_to_n],
            dones=self.dones[r[:, None], t_to_n],
            firststeps=self.firststeps[r, t],
            weights=weights)

    def _check_memory(self, rollout: NamedTuple):
        shapes = {k: v.shape[1:] for k, v in rollout._asdict().items()} # remove n_envs
        rollout_bytes = sum(np.prod(shape) for shape in shapes.values())
        total_bytes = self.capacity  * rollout_bytes
        if psutil is not None:
            avail_bytes = psutil.virtual_memory().available
        if total_bytes > avail_bytes:
            avail_bytes /= 1024 ** 3
            total_bytes /= 1024 ** 3
            warnings.warn(
                """This system does not have enough memory to store the replay buffer.
                Available memory: {avail_bytes:.2f} GB
                Required memory: {total_bytes:.2f} GB
                Difference: {avail_bytes - total_bytes:.2f} GB"""
            )


class ThreadSafeReplayBuffer(ReplayBuffer):
    """ Implements lock on all class methods and timers """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self.lock_add_time = deque(maxlen=10)
        self.add_time = deque(maxlen=10)
        self.lock_sample_time = deque(maxlen=10)
        self.sample_time = deque(maxlen=10)
        self.lock_update_priorities_time = deque(maxlen=10)
        self.update_priorities_time = deque(maxlen=10)

    def thread_safe_add(self, *args, **kwargs):
        lock_add_time_start = time.time()
        with self.lock:
            add_time_start = time.time()
            self.add(*args, **kwargs)
            self.add_time.append(time.time() - add_time_start)
        self.lock_add_time.append(time.time() - lock_add_time_start)

    def thread_safe_sample(self, *args, **kwargs):
        lock_sample_time_start = time.time()
        with self.lock:
            sample_time_start = time.time()
            self.sample_time.append(time.time() - sample_time_start)
        self.lock_sample_time.append(time.time() - lock_sample_time_start)
        return self.sample(*args, **kwargs)

    def thread_safe_update_priorities(self, *args, **kwargs):
        lock_update_priorities_start = time.time()
        with self.lock:
            update_priorities_time_start = time.time()
            self.update_priorities(*args, **kwargs)
            self.update_priorities_time.append(time.time() - update_priorities_time_start)
        self.lock_update_priorities_time.append(time.time() - lock_update_priorities_start)

    def thread_safe_ready(self, min_size):
        with self.lock:
            return self.ready(min_size)


class QNetwork(nn.Module):
    action_dim: int
    scale: float = 1 / jnp.sqrt(2)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        x = self.scale_gradient(x, self.scale)

        # value stream
        v = nn.Dense(512)(x)
        v = nn.relu(v)
        value = nn.Dense(1)(v)
        
        # adv stream
        a = nn.Dense(512)(x)
        a = nn.relu(a)
        adv = nn.Dense(self.action_dim)(a)
        
        q_values = value + (adv - jnp.mean(adv, axis=1, keepdims=True))
        return q_values
    
    def scale_gradient(self, x: jnp.ndarray, scale: float) -> jnp.ndarray:
        """Multiplies the gradient of `x` by `scale`."""
        @jax.custom_gradient
        def wrapped(x: jnp.ndarray):
            return x, lambda grad: (grad * scale,)
        return wrapped(x)


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class Transition(NamedTuple):
    obs: list
    q_values: list
    actions: list
    rewards: list
    dones: list
    firststeps: list


@jax.jit
def calculate_priorities(rollout, q_state):
    # rollout.shape (n_envs, rollout_length, ...)
    n = args.bootstrap_length
    q_tm1 = rollout.q_values[:, :-n]
    a_tm1 = rollout.actions[:, :-1].astype(jnp.int32)
    r_t = rollout.rewards[:, :-1]
    dones_t = rollout.dones[:, :-1]
    obs_tpn = rollout.obs[:, n:]

    apply = jax.vmap(q_network.apply, in_axes=(None, 0))
    q_tpn_val = apply(q_state.target_params, obs_tpn)
    q_tpn_select = apply(q_state.params, obs_tpn)

    @jax.vmap
    def calculate_priorities_over_envs_dim(_q_tm1, _a_tm1, _r_t, _dones_t, _q_tpn_val, _q_tpn_select): 
        i_range = jnp.arange(args.rollout_length)

        @jax.vmap
        def calculate_priorities_over_rollout_dim(i):
            q_tm1_i = _q_tm1[i]
            a_tm1_i = _a_tm1[i]
            r_tn_i = jax.lax.dynamic_slice_in_dim(_r_t, i, n)
            dones_tn_i = jax.lax.dynamic_slice_in_dim(_dones_t, i, n)
            q_tpn_val_i = _q_tpn_val[i]
            q_tpn_select_i = _q_tpn_select[i]
            return multi_step_double_q_learning(
                q_tm1=q_tm1_i,
                a_tm1=a_tm1_i,
                r_tn=r_tn_i,
                dones_tn=dones_tn_i,
                q_tpn_value=q_tpn_val_i,
                q_tpn_selector=q_tpn_select_i,
            )
        return calculate_priorities_over_rollout_dim(i_range)

    td_error = calculate_priorities_over_envs_dim(
        q_tm1, a_tm1, r_t, dones_t, q_tpn_val, q_tpn_select
    )

    return jnp.abs(td_error).reshape(args.n_envs, args.rollout_length)

def local_buffer_to_replay_buffer(queue, rb):
    priority_storage = []
    rollout_storage = []
    while True:
        buffer, q_state, device = queue.get()
        rollout = jax.tree_map(lambda *xs: np.stack(xs).swapaxes(0, 1), *buffer) # [n_envs, rollout_length, ...]
        d_rollout = jax.tree_map(lambda x: jax.device_put(x, device), rollout)
        priorities = calculate_priorities(d_rollout, q_state)
        priorities = np.array(priorities)
        priority_storage.append(priorities)
        rollout_storage.append(rollout)

        if len(rollout_storage) == args.num_actor_threads * len(args.actor_device_ids):
            priorities = jax.tree_map(lambda *xs: np.concatenate(xs), *priority_storage)
            rollouts = jax.tree_map(lambda *xs: np.concatenate(xs), *rollout_storage)
            rb.thread_safe_add(priorities, rollouts)
            priority_storage = []
            rollout_storage = []

def epilson_values_fn(N):
    i_range = jnp.arange(N)
    epsilon_values = args.epsilon ** (1 + (i_range / (N - 1)) * args.epsilon_greedy_alpha)
    return jnp.expand_dims(epsilon_values, axis=-1)

@jax.jit
def get_action(params, obs, epsilon, rng):
    q_values = q_network.apply(params, obs)
    action = distrax.EpsilonGreedy(q_values, epsilon).sample(seed=rng)
    return q_values, action

def rollout(
    args,
    rb,
    params_queue,
    writer,
    thread_id,
    actor_device,
    local_queue,
    ratio_store,
):
    print(f"Actor {thread_id} has started on device {actor_device}")
    envs = make_env(args.env_id, args.seed, args.n_envs)()
    epsilon_values = epilson_values_fn(args.n_envs)

    key = jax.random.PRNGKey(args.seed)

    episode_returns = np.zeros((args.n_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.n_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.n_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.n_envs,), dtype=np.float32)
    running_greedy_episodic_return = deque(maxlen=20)

    storage = []
    local_buffer: list[Transition] = []
    envs.async_reset()
    data_generation_start = time.time()
    actor_network_version = 0
    for step in range(args.total_timesteps):
        global_step = step * args.num_actor_threads * len(args.actor_device_ids) * args.n_envs
        key, subkey = jax.random.split(key)
        if len(params_queue.queue) > 0:
            q_state = params_queue.get()
            actor_network_version += 1

        next_obs, rewards, dones, info = envs.recv()
        truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
        firststeps = info['elapsed_step'] == 0
        env_id = info["env_id"]

        d_next_obs = jax.device_put(next_obs, actor_device)
        d_epsilon_values = jax.device_put(epsilon_values, actor_device)
        d_subkey = jax.device_put(subkey, actor_device)
        d_next_q_values, d_next_actions = get_action(q_state.params, d_next_obs, d_epsilon_values, d_subkey)
        next_q_values = np.array(d_next_q_values)
        next_actions = np.array(d_next_actions)

        envs.send(next_actions)

        # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
        # so we use our own truncated flag
        truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
        episode_returns[env_id] += info["reward"]
        returned_episode_returns[env_id] = np.where(
            info["terminated"] + truncated, episode_returns[env_id], returned_episode_returns[env_id]
        )
        if info['terminated'][-1] or truncated[-1]:
            running_greedy_episodic_return.append(episode_returns[-1])
        episode_returns[env_id] *= (1 - info["terminated"]) * (1 - truncated)
        episode_lengths[env_id] += 1
        returned_episode_lengths[env_id] = np.where(
            info["terminated"] + truncated, episode_lengths[env_id], returned_episode_lengths[env_id]
        )
        episode_lengths[env_id] *= (1 - info["terminated"]) * (1 - truncated)

        if thread_id == 0:
            ratio_store['actor_sps'] = global_step / (time.time() - data_generation_start)
            if step % args.log_interval == 0 and step > 0:
                print(f"global_step={global_step}, SPS: {int(ratio_store['actor_sps'])} episodic_return={np.mean(returned_episode_returns):.2f}, episodic_length={returned_episode_lengths.mean():.2f}")
                writer.add_scalar("charts/actor/episodic_return", np.mean(returned_episode_returns), global_step)
                writer.add_scalar("charts/actor/episodic_length", np.mean(returned_episode_lengths), global_step)
                writer.add_scalar("charts/actor/max_episodic_return", np.max(returned_episode_returns), global_step)
                writer.add_scalar("charts/actor/argmax_episodic_return", np.argmax(returned_episode_lengths), global_step)
                if len(running_greedy_episodic_return) != 0:
                    writer.add_scalar("charts/actor/avg_greedy_episodic_return", np.mean(running_greedy_episodic_return), global_step)
                    writer.add_scalar("charts/actor/greedy_episodic_return", running_greedy_episodic_return[-1], global_step)

        if len(storage) != 0:
            obs, q_values, actions, last_env_ids = storage.pop()
            assert np.all(last_env_ids == env_id)
            local_buffer.append(
                Transition(obs, q_values, actions, rewards, dones, firststeps)
            )

        if len(local_buffer) == args.rollout_length + args.bootstrap_length:
            local_queue.put((local_buffer, q_state, actor_device))
            local_buffer = local_buffer[-1:]

        storage.append((next_obs, next_q_values, next_actions, env_id))

        while rb.thread_safe_ready(args.learning_starts) and ratio_store['actor_sps'] / ratio_store['learner_sps'] > 800:
            time.sleep(0.01)


if __name__ == "__main__":
    args = Args()
    
    devices = jax.devices()
    actor_devices = [devices[i] for i in args.actor_device_ids]
    learner_devices = [devices[i] for i in args.learner_device_ids]
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    envs = make_env(args.env_id, args.seed, args.n_envs)()
    q_network = QNetwork(action_dim=envs.single_action_space.n)
    obs = jnp.zeros(envs.single_observation_space.shape, dtype=jnp.float32)[None,]
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    q_state = flax.jax_utils.replicate(q_state, learner_devices)

    rb = ThreadSafeReplayBuffer(
        capacity=args.buffer_size,
        length=args.rollout_length,
        bootstrap_n=args.bootstrap_length,
        n_envs=args.n_envs * len(args.actor_device_ids) * args.num_actor_threads,
        seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
    )

    def multi_step_double_q_learning(q_tm1, a_tm1, r_tn, dones_tn, q_tpn_value, q_tpn_selector):
        """
        Calculates the multi-step double Q-learning TD error.

        Args:
            q_tm1: Q-values at time t-1
            a_tm1: actions at time t-1
            r_tn: rewards at time [t, t+1, ..., t+n]
            dones_tn: dones at time [t, t+1, ..., t+n]
            q_tpn_value: Q-values at time t+n
            q_tpn_selector: Q-values at time t+n
        """
        # both r_tpn and q_tpn are in the multi-step return
        pad_dones_tn = jnp.concatenate([dones_tn, jnp.zeros_like(dones_tn[-1:])])
        
        # "multi-step returns are truncated if the episode ends in fewer than n steps" - Horgan et al. 2018 page 4
        for i in range(pad_dones_tn.shape[0]):
            new_value = pad_dones_tn[i] + pad_dones_tn[i+1] > 0
            pad_dones_tn = pad_dones_tn.at[i+1].set(new_value)
        
        n = r_tn.shape[0]
        n_range = jnp.arange(n+1)
        gamma_tn = (1 - pad_dones_tn) * jnp.power(args.gamma, n_range)
        q_target = q_tpn_value[q_tpn_selector.argmax()]
        target_tm1 = jnp.sum(gamma_tn * jnp.concatenate([r_tn, q_target[None,]]))
        target_tm1 = jax.lax.stop_gradient(target_tm1)

        return target_tm1 - q_tm1[a_tm1]

    def apex_dqn_loss(online_params, target_params, obs_tm1, a_tm1, obs_tpn, r_tn, dones_tn, firststeps, weights):
        mask = 1 - firststeps # TODO: determine if info['elapsed_step'] are invalid "dummy" steps 
        q_tm1 = q_network.apply(online_params, obs_tm1)
        q_tpn_val = q_network.apply(target_params, obs_tpn)
        q_tpn_select = q_network.apply(online_params, obs_tpn)
        td_error = jax.vmap(multi_step_double_q_learning)(
            q_tm1=q_tm1,
            a_tm1=a_tm1.astype(jnp.int32),
            r_tn=r_tn,
            dones_tn=dones_tn,
            q_tpn_value=q_tpn_val,
            q_tpn_selector=q_tpn_select,
        )
        weighted_td_error = (weights * td_error) * mask
        return jnp.mean(weighted_td_error ** 2), (q_tm1, jnp.abs(td_error)) # TODO: try rlax.l2_loss (0.5 *) can significanlty change loss landscape

    def single_device_update(q_state, batch):
        value_and_grad = jax.value_and_grad(apex_dqn_loss, has_aux=True)
        (loss_value, aux), grads = value_and_grad(
            q_state.params,
            q_state.target_params,
            batch.obs,
            batch.actions,
            batch.bootstrap_obs,
            batch.rewards,
            batch.dones,
            batch.firststeps,
            batch.weights,
        )
        q_pred, priorities = aux
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, priorities, q_state

    multi_device_update = jax.pmap(single_device_update, axis_name="local_devices", devices=learner_devices)

    actor_to_buffer_queue = queue.Queue(maxsize=args.num_actor_threads * len(args.actor_device_ids))
    threading.Thread(target=local_buffer_to_replay_buffer, args=(actor_to_buffer_queue, rb)).start()

    from types import SimpleNamespace
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    ratio_store = {"actor_sps": 1e-6,"learner_sps": 1e-6}
    param_queues = []
    unreplicated_q_state = flax.jax_utils.unreplicate(q_state)
    for device_id, device  in enumerate(actor_devices):
        device_params = jax.device_put(unreplicated_q_state, device)
        for thread_id in range(args.num_actor_threads):
            global_step_queue = queue.Queue()
            params_queue = queue.Queue()
            params_queue.put(device_params)
            to_use_writer = dummy_writer if thread_id > 0 else writer
            threading.Thread(
                target=rollout,
                args=(
                    args,
                    rb,
                    params_queue,
                    to_use_writer,
                    device_id * args.num_actor_threads + thread_id,
                    device,
                    actor_to_buffer_queue,
                    ratio_store
                    )
                ).start()
            param_queues.append(params_queue)

    print(f"{args.n_envs * len(args.actor_device_ids) * args.num_actor_threads} environments have started")

    # don't start training until replay buffer is sufficiently full
    while not rb.thread_safe_ready(args.learning_starts):
        time.sleep(1)

    batch_queue = queue.Queue(maxsize=args.n_prefetch)
    def prefetch_batches(rb, queue, learner_device):
        sample_batch_time = deque(maxlen=10)
        device_put_time = deque(maxlen=10)
        while True:
            sample_batch_time_start = time.time()
            batch, inds = rb.thread_safe_sample(args.batch_size)
            sample_batch_time.append(time.time() - sample_batch_time_start)
            device_put_time_start = time.time()
            batch = jax.tree_map(lambda x: jax.device_put(x, device=learner_device), batch)
            device_put_time.append(time.time() - device_put_time_start)
            queue.put((batch, inds, np.mean(sample_batch_time), np.mean(device_put_time)))

    threading.Thread(target=prefetch_batches, args=(rb, batch_queue, learner_devices[0])).start()
    
    priority_queue = queue.Queue(maxsize=args.n_prefetch)
    def update_priorities(rb, queue):
        while True:
            updated_priorities, inds = queue.get()
            updated_priorities = updated_priorities.reshape(-1) # TODO: make sure order is preserved CRITICAL (you can pass a range and see if it is recovered on the way out)
            updated_priorities = np.array(updated_priorities)
            rb.thread_safe_update_priorities(inds, updated_priorities)
    threading.Thread(target=update_priorities, args=(rb, priority_queue)).start()

    @jax.jit
    def split_data(data):
        return jax.tree_map(lambda x: jnp.split(x, len(learner_devices)), data)

    args.total_train_steps = args.total_timesteps // args.batch_size
    print(f"Starting training for {args.total_train_steps} steps")
    start_time = time.time()
    train_step = 0
    get_batch_time = deque(maxlen=10)
    update_time = deque(maxlen=10)
    while True:
        train_step += 1

        # get a batch
        get_batch_time_start = time.time()
        data, inds, sample_time, device_put_time = batch_queue.get()
        partitioned_batch = split_data(data)
        sharded_batch = Batch(
            *list(map(lambda x: jax.device_put_sharded(x, devices=learner_devices), partitioned_batch))
        )
        get_batch_time.append(time.time() - get_batch_time_start)

        # perform gradient update
        update_time_start = time.time()
        loss, q_value, updated_priorities, q_state = multi_device_update(
            q_state,
            sharded_batch,
        )
        update_time.append(time.time() - update_time_start)
        priority_queue.put((updated_priorities, inds))

        # put the updated params to the actors
        unreplicated_q_state = flax.jax_utils.unreplicate(q_state)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_q_state, devices[d_id])
            for thread_id in range(args.num_actor_threads):
                param_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

        ratio_store['learner_sps'] = train_step / (time.time() - start_time)
        ratio = ratio_store['actor_sps'] / ratio_store['learner_sps']
        if train_step % 100 == 0:
            writer.add_scalar("losses/td_loss", jax.device_get(loss).mean(), train_step)
            writer.add_scalar("losses/q_values", jax.device_get(q_value).mean(), train_step)
            writer.add_scalar("charts/learner/SPS", train_step / (time.time() - start_time), train_step)
            writer.add_scalar("charts/learner/get_batch_time", np.mean(get_batch_time), train_step)
            writer.add_scalar("charts/learner/sample_batch_time", sample_time, train_step)
            writer.add_scalar("charts/learner/device_put_time", device_put_time, train_step)
            writer.add_scalar("charts/learner/update_time", np.mean(update_time), train_step)
            print(f"TSPS: {int(ratio_store['learner_sps'])}, ratio {ratio:.2f}, q_value {jax.device_get(q_value).mean():.3f} step {train_step} update time {np.mean(update_time):.3f} batch_queue size {batch_queue.qsize()}")
            # replay buffer metrics
            writer.add_scalar("charts/buffer/add_time", np.mean(rb.add_time), train_step)
            writer.add_scalar("charts/buffer/add_time_delta", np.mean(rb.lock_add_time) - np.mean(rb.add_time), train_step)
            writer.add_scalar("charts/buffer/sample_time", np.mean(rb.sample_time), train_step)
            writer.add_scalar("charts/buffer/sample_time_delta", np.mean(rb.lock_sample_time) - np.mean(rb.sample_time), train_step)
            writer.add_scalar("charts/buffer/update_priorities_time", np.mean(rb.update_priorities_time), train_step)
            writer.add_scalar("charts/buffer/update_priorities_time_delta", np.mean(rb.lock_update_priorities_time) - np.mean(rb.update_priorities_time), train_step)

        if train_step % args.target_network_frequency == 0:
            q_state = q_state.replace(target_params=q_state.params)

        if train_step == args.total_train_steps:
            break

    envs.close()
    writer.close()
    
    sys.exit()