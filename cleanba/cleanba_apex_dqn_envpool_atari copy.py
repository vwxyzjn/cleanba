# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import sys
import tyro
import random
import time
import queue
import threading
from dataclasses import dataclass
from typing import NamedTuple
from collections import deque
from types import SimpleNamespace

import rlax
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import envpool
from tensorboardX import SummaryWriter

import psutil
import warnings
from typing import NamedTuple

import numpy as np

f32 = np.float32

class Batch(NamedTuple):
    obs: np.ndarray
    bootstrap_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    firststeps: np.ndarray
    weights: np.ndarray

class PERReplayBuffer:
    """
    Replay buffer for storing rollouts of trajectories.
    
    :param capacity: maximum number of rollouts to store
    :param length: length of each rollout
    :param valid_length: number of timesteps in each rollout that can be sampled
    :param n_envs: number of parallel environments
    :param seed: random seed
    :param alpha: exponent for prioritized sampling (0 = uniform sampling)
    :param beta: exponent for importance sampling (0 = no correction)
    
    """
    def __init__(
        self, capacity: int, length: int, valid_length: int,
        n_envs: int, seed: int = 0, alpha: float = 1., beta: float = 1.,
        ) -> None:
        self.capacity = capacity
        self.length = length
        self.valid_length = valid_length
        self.n = self.length - self.valid_length # bootstrap length
        self.n_envs = n_envs
        self.pos = 0
        self.is_setup = False
        self.rng = np.random.default_rng(seed)
        
        self.alpha = alpha
        self.beta = beta
        
        self.is_setup = False
        self.full = False
        self.timesteps_seen = 0

        assert self.capacity % self.n_envs == 0, "Capacity must be evenly divisible by n_envs"
        # TODO: remove via reversing the indexing method (i.e. [self.pos-length:self.pos])

    def _setup(self, example: NamedTuple):
        assert len(example.rewards.shape) == 2, "Rewards must be scalars (i.e. shape (n_envs, length,)))"
        shapes = {k: v.shape[2:] for k, v in example._asdict().items()}
        dtype = {k: v.dtype for k, v in example._asdict().items()}
        self._check_memory(example)

        self.priorities = np.zeros((self.capacity, self.valid_length), dtype=f32)
        self.obs = np.zeros((self.capacity, self.length, *shapes['obs']), dtype=dtype['obs'])
        self.actions = np.zeros((self.capacity, self.length, *shapes['actions']), dtype=dtype['actions'])
        self.rewards = np.zeros((self.capacity, self.length, *shapes['rewards']), dtype=dtype['rewards'])
        self.dones = np.zeros((self.capacity, self.length, *shapes['dones']), dtype=dtype['dones'])
        self.firststeps = np.zeros((self.capacity, self.length, *shapes['firststeps']), dtype=dtype['firststeps'])

        self.is_setup = True

    def add(self, priorities: np.ndarray, rollout: NamedTuple):
        assert priorities.shape == (self.n_envs, self.valid_length), "Priorities must be shape (n_envs, valid_length)"
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
        assert self.is_setup, "Replay buffer must be setup before sampling"
        inds, weights = self._priority_sampling(batch_size)
        return self._get_samples(inds, weights), inds

    def _priority_sampling(self, batch_size: int):
        N = self.capacity * self.valid_length # (number of rollouts * indexable length)

        sample_probs = self.priorities.flatten() ** self.alpha
        sample_probs /= sample_probs.sum() 
        sampled_inds = self.rng.choice(N, batch_size, p=sample_probs, replace=False)

        importance_sampling_weights = (N * sample_probs[sampled_inds]) ** -self.beta
        importance_sampling_weights /= importance_sampling_weights.max()
        return sampled_inds, importance_sampling_weights

    def _get_samples(self, inds, weights: np.ndarray):
        r, t = np.unravel_index(inds, self.priorities.shape) # rollout index, transition index
        btsrp = t + self.n # next transition
        n_range = np.arange(self.n)
        t_to_n = t[:, None] + n_range[None,] # transition to bootstrap transition
        batch = Batch(
            obs=self.obs[r, t],
            bootstrap_obs=self.obs[r, btsrp],
            actions=self.actions[r, t],
            rewards=self.rewards[r[:, None], t_to_n],
            dones=self.dones[r[:, None], t_to_n],
            firststeps=self.firststeps[r, t],
            weights=weights,
        )
        return batch
    
    def update_priorities(self, inds: np.ndarray, updated_priorities: np.ndarray):
        b, t = np.unravel_index(inds, self.priorities.shape)
        self.priorities[b, t] = updated_priorities
    
    def size(self):
        return self.pos if not self.full else self.capacity
    
    def ready(self, min_size: int):
        return self.timesteps_seen > min_size
    
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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    "the name of this experiment"
    seed: int = 1
    "the seed of the experiment"
    track: bool = True
    "if toggled, this experiment will be tracked with Weights and Biases"
    wandb_project_name: str = "Distributed PER"
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
    n_envs: int = 100
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
    learning_starts: int = 50_000 # TODO 
    "the timestep it takes to start learning"
    n_prefetch: int = 16
    "the number of batches to prefetch for learner"
    log_interval: int = 100
    "the log interval"

    actor_device_ids = [0, 1]#[0,1,2,3]
    "actor devices"
    learner_device_ids = [4,5,6,7]
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

def scale_gradient(x: jnp.ndarray, scale: float) -> jnp.ndarray:
    """Multiplies the gradient of `x` by `scale`."""
    @jax.custom_gradient
    def wrapped(x: jnp.ndarray):
        return x, lambda grad: (grad * scale,)
    return wrapped(x)


class QNetwork(nn.Module):
    action_dim: int
    scale_factor = 1 / jnp.sqrt(2)

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
        
        # x = scale_gradient(x, self.scale_factor)

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

class TrainState(TrainState):
    target_params: flax.core.FrozenDict

def epilson_values_fn(N):
    i_range = jnp.arange(N)
    epsilon_values = args.epsilon ** (1 + (i_range / (N - 1)) * args.epsilon_greedy_alpha)
    return jnp.expand_dims(epsilon_values, axis=-1)

class ThreadSafeReplayBuffer(PERReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def thread_safe_add(self, *args, **kwargs):
        with self.lock:
            self.add(*args, **kwargs)

    def thread_safe_sample(self, *args, **kwargs):
        with self.lock:
            return self.sample(*args, **kwargs)

    def thread_safe_update_priorities(self, *args, **kwargs):
        with self.lock:
            self.update_priorities(*args, **kwargs)

    def thread_safe_ready(self, min_size):
        with self.lock:
            return self.ready(min_size)


class Transition(NamedTuple):
    obs: list
    q_values: list
    actions: list
    rewards: list
    dones: list
    firststeps: list


@jax.jit
def calculate_priorities(rollout, q_state):
    # rollout: [n_envs, rollout_length, ...]
    n = args.bootstrap_length
    q_tm1 = rollout.q_values[:, :-n]
    a_tm1 = rollout.actions[:, :-1].astype(jnp.int32)
    r_t = rollout.rewards[:, :-1]
    dones_t = rollout.dones[:, :-1]
    obs_tpn = rollout.obs[:, n:]
    
    apply = jax.vmap(q_network.apply, in_axes=(None, 0))
    q_tpn_val = apply(q_state.target_params, obs_tpn)
    q_tpn_select = apply(q_state.params, obs_tpn)
    
    def over_n_envs_multi_step_double_q_learning(_q_tm1, _a_tm1, _r_t, _dones_t, _q_tpn_val, _q_tpn_select): 
        i_range = jnp.arange(args.valid_length)
        def vectorize_over_rollout_index(i):
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
        # vectorize over valid indicies in the rollout
        return jax.vmap(vectorize_over_rollout_index)(i_range)

    # vmap over n_envs
    vectorized_multi_step_double_q_learning = jax.vmap(over_n_envs_multi_step_double_q_learning)
    td_error = vectorized_multi_step_double_q_learning(
        q_tm1, a_tm1, r_t, dones_t, q_tpn_val, q_tpn_select
    )

    return jnp.abs(td_error).reshape(args.n_envs, args.valid_length)


def local_buffer_to_replay_buffer(queue, rb, device, queue_time, make_rollout_time, calculate_priorities_time, rb_add_time):
    while True:
        queue_time_start = time.time()
        buffer, q_state = queue.get()
        queue_time.append(time.time() - queue_time_start)
        make_rollout_time_start = time.time()
        rollout = jax.tree_map(lambda *xs: np.asarray(xs).swapaxes(0,1), *buffer) # asarray may introduce silent bug
        d_rollout = jax.tree_map(lambda x: jax.device_put(x, device), rollout)
        make_rollout_time.append(time.time() - make_rollout_time_start)
        calculate_priorities_time_start = time.time()
        priorities = calculate_priorities(d_rollout, q_state) # possible q_state isn't on the correct device?
        cpu_priorities = jax.device_get(priorities)
        calculate_priorities_time.append(time.time() - calculate_priorities_time_start)
        rb_add_time_start = time.time()
        rb.thread_safe_add(cpu_priorities, rollout)
        rb_add_time.append(time.time() - rb_add_time_start)

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
    sps_store,
):
    print(f"Actor {thread_id} has started on device {actor_device}")
    envs = make_env(args.env_id, args.seed, args.n_envs)()
    epsilon_values = epilson_values_fn(args.n_envs)

    env_recv_time = deque(maxlen=10)
    get_action_h2d_time = deque(maxlen=10)
    get_action_time = deque(maxlen=10)
    get_action_d2h_time = deque(maxlen=10)
    env_send_time = deque(maxlen=10)
    storage_time = deque(maxlen=10)
    local_buffer_get_time = deque(maxlen=10)
    make_rollout_time = deque(maxlen=10)
    calculate_priorities_time = deque(maxlen=10)
    replay_buffer_add_time = deque(maxlen=10)

    key = jax.random.PRNGKey(args.seed)
    local_queue = queue.Queue(maxsize=args.num_actor_threads * len(args.actor_device_ids))
    threading.Thread(
        target=local_buffer_to_replay_buffer,
        args=(
            local_queue,
            rb,
            actor_device,
            local_buffer_get_time,
            make_rollout_time,
            calculate_priorities_time,
            replay_buffer_add_time)
        ).start()

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

        env_recv_time_start = time.time()
        next_obs, rewards, dones, info = envs.recv()
        env_recv_time.append(time.time() - env_recv_time_start)
        truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
        firststeps = info['elapsed_step'] == 0
        env_id = info["env_id"]

        get_action_h2d_time_start = time.time()
        d_next_obs = jax.device_put(next_obs, actor_device)
        d_epsilon_values = jax.device_put(epsilon_values, actor_device)
        d_subkey = jax.device_put(subkey, actor_device)
        get_action_h2d_time.append(time.time() - get_action_h2d_time_start)
        get_action_time_start = time.time()
        next_q_values, next_actions = get_action(q_state.params, d_next_obs, d_epsilon_values, d_subkey)
        get_action_d2h_time_start = time.time()
        next_actions = jax.device_get(next_actions)
        get_action_d2h_time.append(time.time() - get_action_d2h_time_start)
        get_action_time.append(time.time() - get_action_time_start)

        env_send_time_start = time.time()
        envs.send(next_actions)
        env_send_time.append(time.time() - env_send_time_start)

        # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
        # so we use our own truncated flag
        truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
        episode_returns[env_id] += info["reward"]
        returned_episode_returns[env_id] = np.where(
            info["terminated"] + truncated, episode_returns[env_id], returned_episode_returns[env_id]
        )
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if info['terminated'][-1] or truncated[-1]:
            running_greedy_episodic_return.append(episode_returns[-1])
        episode_returns[env_id] *= (1 - info["terminated"]) * (1 - truncated)
        episode_lengths[env_id] += 1
        returned_episode_lengths[env_id] = np.where(
            info["terminated"] + truncated, episode_lengths[env_id], returned_episode_lengths[env_id]
        )
        episode_lengths[env_id] *= (1 - info["terminated"]) * (1 - truncated)

        learner_sps = int(global_step / (time.time() - data_generation_start))
        sps_store.append((learner_sps, global_step))

        storage_time_start = time.time()
        if len(storage) != 0:
            obs, q_values, actions, last_env_ids = storage.pop()
            assert np.all(last_env_ids == env_id)
            local_buffer.append(
                Transition(obs, q_values, actions, rewards, dones, firststeps)
            )
        storage_time.append(time.time() - storage_time_start)

        if len(local_buffer) == args.rollout_length:
            local_queue.put((local_buffer, q_state))
            local_buffer = local_buffer[-1:]

        # "next" indicates that the variable corresponds to the next iteration of the loop
        storage.append((next_obs, next_q_values, next_actions, env_id))

        if step % args.log_interval == 0 and step > 0 and thread_id == 0:
            print(f"global_step={global_step}, SPS: {learner_sps} episodic_return={np.mean(returned_episode_returns):.2f}, episodic_length={returned_episode_lengths.mean():.2f}")
            writer.add_scalar("charts/episodic_return", np.mean(returned_episode_returns), global_step)
            writer.add_scalar("charts/episodic_length", np.mean(returned_episode_lengths), global_step)
            writer.add_scalar("charts/max_episodic_return", np.max(returned_episode_returns), global_step)
            writer.add_scalar("charts/argmax_episodic_return", np.argmax(returned_episode_lengths), global_step)
            if len(running_greedy_episodic_return) != 0:
                writer.add_scalar("charts/avg_greedy_episodic_return", np.mean(running_greedy_episodic_return), global_step)
                writer.add_scalar("charts/greedy_episodic_return", running_greedy_episodic_return[-1], global_step)
            writer.add_scalar("stats/actor/env_recv_time", np.mean(env_recv_time), global_step)
            writer.add_scalar("stats/actor/get_action_h2d_time", np.mean(get_action_h2d_time), global_step)
            writer.add_scalar("stats/actor/get_action_time", np.mean(get_action_time), global_step)
            writer.add_scalar("stats/actor/get_action_d2h_time", np.mean(get_action_d2h_time), global_step)
            writer.add_scalar("stats/actor/storage_time", np.mean(storage_time), global_step)
            if rb.thread_safe_ready(args.learning_starts):
                writer.add_scalar("stats/actor/local_buffer_get_time", np.mean(local_buffer_get_time), global_step)
                writer.add_scalar("stats/actor/make_rollout_time", np.mean(make_rollout_time), global_step)
                writer.add_scalar("stats/actor/calculate_priorities_time", np.mean(calculate_priorities_time), global_step)
                writer.add_scalar("stats/actor/replay_buffer_add_time", np.mean(replay_buffer_add_time), global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
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

    args.rollout_length = args.rollout_length + args.bootstrap_length
    args.valid_length = args.rollout_length - args.bootstrap_length
    rb = ThreadSafeReplayBuffer(
        capacity=args.buffer_size,
        length=args.rollout_length,
        valid_length=args.valid_length,
        n_envs=args.n_envs,
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
            r_tn: rewards at time t, t+1, ..., t+n
            dones_tn: dones at time t, t+1, ..., t+n
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

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=learner_devices
    )

    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None
    actor_sps_store = deque(maxlen=10)

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
                args=(args, rb, params_queue, to_use_writer, device_id * args.num_actor_threads + thread_id, device, actor_sps_store)
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
    
    priorities_queue = queue.Queue(maxsize=args.n_prefetch)
    def update_priorities(rb, queue):
        while True:
            updated_priorities = queue.get()
            update_priorities_time_start = time.time()
            rb.thread_safe_update_priorities(inds, updated_priorities)
            update_priorities_time.append(time.time() - update_priorities_time_start)

    threading.Thread(target=update_priorities, args=(rb, priorities_queue)).start()

    @jax.jit
    def split_data(data):
        return jax.tree_map(lambda x: jnp.split(x, len(learner_devices)), data)

    args.total_train_steps = args.total_timesteps // args.batch_size
    print(f"Starting training for {args.total_train_steps} steps")
    start_time = time.time()
    learner_network_version = 0
    get_batch_time = deque(maxlen=10)
    update_time = deque(maxlen=10)
    update_priorities_time = deque(maxlen=10)
    put_params_time = deque(maxlen=10)
    while True:
        learner_network_version += 1

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
        updated_priorities = updated_priorities.reshape(-1)
        cpu_updated_priorities = np.array(updated_priorities) # you can remove this and make the transfer device op happen in buffer preventing from blocking training 
        update_time.append(time.time() - update_time_start)
        loss = loss.mean(0)
        q_value = q_value.mean(0)

        priorities_queue.put(cpu_updated_priorities)

        # put the updated params to the actors
        put_params_time_start = time.time()
        unreplicated_q_state = flax.jax_utils.unreplicate(q_state)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_q_state, devices[d_id])
            for thread_id in range(args.num_actor_threads):
                param_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)
        put_params_time.append(time.time() - put_params_time_start)

        actor_sps, global_step = actor_sps_store[0]
        learner_sps = learner_network_version / (time.time() - start_time)
        if learner_network_version % 100 == 0:
            writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
            writer.add_scalar("losses/q_values", jax.device_get(q_value).mean(), global_step)
            writer.add_scalar("stats/ratio/actor_sps", actor_sps, global_step)
            writer.add_scalar("stats/ratio/learner_sps", learner_sps, global_step)
            writer.add_scalar("stats/learner/get_batch_time", np.mean(get_batch_time), global_step)
            writer.add_scalar("stats/learner/sample_batch_time", sample_time, global_step)
            writer.add_scalar("stats/learner/device_put_time", device_put_time, global_step)
            writer.add_scalar("stats/learner/update_time", np.mean(update_time), global_step)
            writer.add_scalar("stats/learner/update_priorities_time", np.mean(update_priorities_time), global_step)
            writer.add_scalar("stats/learner/put_params_time", np.mean(put_params_time), global_step)
            writer.add_scalar("stats/learner/batch_queue_size", batch_queue.qsize(), global_step)
            writer.add_scalar("charts/learner_network_version", learner_network_version, global_step)
            print(f"TSPS: {int(learner_network_version / (time.time() - start_time))} q_value {jax.device_get(q_value).mean():.3f} step {learner_network_version} update time {np.mean(update_time):.3f} batch_queue size {batch_queue.qsize()}")

        if learner_network_version % args.target_network_frequency == 0:
            q_state = q_state.replace(target_params=q_state.params)

        if learner_network_version == args.total_train_steps:
            break

    envs.close()
    writer.close()

    sys.exit()