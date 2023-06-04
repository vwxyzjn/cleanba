import os
import queue
import random
import threading
import time
import uuid
from collections import deque
from functools import partial
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, NamedTuple, Optional, Sequence, Tuple

import envpool
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import tyro
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"

f32 = np.float32

class Batch(NamedTuple):
    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    weights: np.ndarray

class ReplayBuffer:
    """
    Replay buffer for storing rollouts of trajectories.
    
    :param capacity: maximum number of rollouts to store
    :param length: length of each rollout
    :param num_steps: number of timesteps in each rollout that can be sampled
    :param n_envs: number of parallel environments
    :param seed: random seed
    :param alpha: exponent for prioritized sampling (0 = uniform sampling)
    :param beta: exponent for importance sampling (0 = no correction)
    
    """
    def __init__(
        self, capacity: int, num_steps: int, boostrap_n: int, n_envs: int, seed: int = 0, alpha: float = 1., beta: float = 1.,
        ) -> None:
        self.capacity = capacity
        self.num_steps = num_steps
        self.n = boostrap_n
        self.length = num_steps + boostrap_n
        self.n_envs = n_envs
        self.pos = 0
        self.is_setup = False
        self.rng = np.random.default_rng(seed)

        self.alpha = alpha
        self.beta = beta

        self.is_setup = False
        self.full = False
        self.timesteps_seen = 0
        
        self.lock = threading.Lock()

        assert self.capacity % self.n_envs == 0, "Capacity must be evenly divisible by n_envs"
        # TODO: remove via reversing the indexing method (i.e. [self.pos-length:self.pos])

    def _setup(self, example: NamedTuple):
        assert len(example.rewards.shape) == 2, "Rewards must be scalars (i.e. shape (n_envs, length,)))"
        shapes = {k: v.shape[2:] for k, v in example._asdict().items()} #[2:] removes time and batch dimensions
        dtypes = {k: v.dtype for k, v in example._asdict().items()}

        self.total_sampleable_indicies = self.capacity * self.num_steps
        self.priorities = np.zeros((self.capacity, self.num_steps), dtype=f32)
        self.obs = np.zeros((self.capacity, self.length, *shapes['obs']), dtype=dtypes['obs'])
        self.rewards = np.zeros((self.capacity, self.length), dtype=dtypes['rewards'])
        self.actions = np.zeros((self.capacity, self.length, *shapes['actions']), dtype=dtypes['actions'])
        self.dones = np.zeros((self.capacity, self.length), dtype=dtypes['dones'])

        self.is_setup = True

    def add(self, priorities: np.ndarray, rollout: NamedTuple):
        with self.lock:
            assert priorities.shape == (self.n_envs, self.num_steps), "Priorities must be shape (n_envs, num_steps)"
            if not self.is_setup:
                self._setup(rollout)
            indexer = slice(self.pos, self.pos + self.n_envs)

            self.priorities[indexer] = np.array(priorities).copy()
            self.obs[indexer] = np.array(rollout.obs).copy()
            self.actions[indexer] = np.array(rollout.actions).copy()
            self.rewards[indexer] = np.array(rollout.rewards).copy()
            self.dones[indexer] = np.array(rollout.dones).copy()

            self.timesteps_seen += self.n_envs * self.num_steps
            self.pos += self.n_envs
            if self.pos >= self.capacity:
                self.full = True
                self.pos = 0

    def sample(self, batch_size: int):
        assert self.is_setup, "Replay buffer must be setup before sampling"
        # inds, weights = self._priority_sampling(batch_size)
        if self.full:
            inds = np.random.randint(1, self.total_sampleable_indicies, size=batch_size)
        else:
            inds = np.random.randint(1, self.pos*self.num_steps, size=batch_size)
        weights = np.ones_like(inds, dtype=f32)
        return self._get_samples(inds, weights), inds

    def _priority_sampling(self, batch_size: int):
        N = self.capacity * self.num_steps # (number of rollouts * indexable length)
        
        sample_probs = self.priorities.flatten() ** self.alpha
        sample_probs /= sample_probs.sum() 
        sampled_inds = self.rng.choice(N, batch_size, p=sample_probs, replace=False)

        importance_sampling_weights = (N * sample_probs[sampled_inds]) ** -self.beta
        importance_sampling_weights /= importance_sampling_weights.max()
        return sampled_inds, importance_sampling_weights

    def _get_samples(self, inds: tuple[np.ndarray, np.ndarray], weights: np.ndarray):
        r, t = np.unravel_index(inds, self.priorities.shape) # rollout index, transition index
        # TODO: ASSERT ALL T == 0
        btsrp = t + self.n # next transition
        # n_range = np.arange(self.n)
        # t_to_n = t[:, None] + n_range[None,] # transition to bootstrap transition
        return Batch(
            obs=self.obs[r, t],
            next_obs=self.obs[r, btsrp],
            actions=self.actions[r, t],
            rewards=self.rewards[r, t],
            dones=self.dones[r, t],
            weights=weights
        )

    def update_priorities(self, inds: np.ndarray, updated_priorities: np.ndarray):
        b, t = np.unravel_index(inds, self.priorities.shape)
        self.priorities[b, t] = updated_priorities

    def size(self):
        return self.pos if not self.full else self.capacity

    def ready(self, min_size: int = None):
        return self.timesteps_seen > min_size or self.full
    

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    "the name of this experiment"
    seed: int = 1
    "seed of the experiment"
    track: bool = False
    "if toggled, this experiment will be tracked with Weights and Biases"
    wandb_project_name: str = "cleanRL"
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
    log_frequency: int = 10
    "the logging frequency of the model performance (in terms of `updates`)"

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    "the id of the environment"
    total_timesteps: int = 50e6
    "total timesteps of the experiments"
    learning_rate: float = 1e-4
    "the learning rate of the optimizer"
    local_num_envs: int = 180
    "the number of parallel game environments per actor thread"
    num_steps: int = 1
    "the number of steps in each rollout"
    bootstrap_n: int = 1
    "the number of steps in n-step boostrapping (n=1 results in vanilla q-learning)"
    buffer_capacity: int = 1e6
    "maximum number of rollouts the buffer can store"
    anneal_lr: bool = True
    "Toggle learning rate annealing for policy and value networks"
    gamma: float = 0.99
    "the discount factor gamma"
    start_e: float = 1.
    "staring epsilon value"
    end_e: float = 0.01
    "ending epsilon value"
    exploration_fraction: float = 0.10
    "percentage of total_timesteps until epsilon is decayed from start_e to end_e"
    train_frequency: int = 2
    "how many rollouts to perform per training step (this acts as the main control for obtaining target ratio)"
    target_ratio: int = 700
    "data generation to gradient update ratio critical to performance"
    ratio_tolerance: int = 50
    "acceptable level of devation from the target ratio"
    target_update_frequency: int = 1000
    "number of gradient updates to perform before copying online params to target params"

    actor_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that actor workers will use"
    num_actor_threads: int = 2
    "the number of actor threads to use"
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that learner workers will use"
    distributed: bool = False
    "whether to use `jax.distirbuted`"
    concurrency: bool = True
    "whether to run the actor and learner concurrently"

    # runtime arguments to be filled in
    async_batch_size: int = 0
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    num_updates: int = 0
    async_update: int = 0
    world_size: int = 0
    local_rank: int = 0
    num_envs: int = 0
    batch_size: int = 0
    num_updates: int = 0
    action_dim: int = 0
    global_learner_decices: Optional[List[str]] = None
    actor_devices: Optional[List[str]] = None
    learner_devices: Optional[List[str]] = None


ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping


def make_env(env_id, seed, num_envs, async_batch_size=1):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            batch_size=async_batch_size,
            episodic_life=False,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 6
            repeat_action_probability=0.25,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12
            noop_max=1,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12 (no-op is deprecated in favor of sticky action, right?)
            full_action_space=True,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) Tab. 5
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


def scale_gradient(x: jnp.ndarray, scale: float= 1 / np.sqrt(2)) -> jnp.ndarray:
    """Multiplies the gradient of `x` by `scale`."""
    @jax.custom_gradient
    def wrapped(x: jnp.ndarray):
        return x, lambda grad: (grad * scale,)
    return wrapped(x)


class Network(nn.Module):
    channels: Sequence[int] = (32, 64, 64)
    kernels: Sequence[int] = (8, 4, 3)
    strides: Sequence[int] = (4, 4, 1)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channel, kernel, stride in zip(self.channels, self.kernels, self.strides):
            x = nn.Conv(channel, kernel_size=(kernel, kernel), stride=(stride, stride), padding="VALID")(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return scale_gradient(x) # # Wang et al. 2016 (Dueling DQN)


class ValueStream(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)


class AdvantageStream(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class DuelingQNetwork(nn.Module):
    action_dim: int
    torso: nn.Module = Network()
    value_stream: nn.Module = ValueStream()
    advantage_stream: nn.Module = AdvantageStream(action_dim)

    @nn.compact
    def __call__(self, x):
        hidden = self.torso(x)
        value = self.value_stream(hidden)
        advantage = self.advantage_stream(hidden)
        return value + (advantage - jnp.mean(advantage, axis=1, keepdims=True))


class TrainState(TrainState):
    target: flax.core.FrozenDict

class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    rewards: list
    truncations: list
    terminations: list
    firststeps: list  # first step of an episode

def epsilon_linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def rollout(
    key: jax.random.PRNGKey,
    args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    actor_device,
    rb,
):
    envs = make_env(
        args.env_id,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs,
        args.async_batch_size,
    )()
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    start_time = time.time()

    @jax.jit
    def get_action(
        params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        epsilon: int,
        key: jax.random.PRNGKey,
    ):
        next_obs = jnp.array(next_obs)
        epsilon = jnp.array(epsilon)
        q_value = get_q_value(params, next_obs)
        batch_dim, action_dim = q_values.shape
        trial = jax.random.randint(key, shape=(batch_dim,), minval=0, maxval=1000) / 1000
        random_action = jax.random.randint(key, shape=(batch_dim,) minval=0, maxval=action_dim)
        greedy_action = q_values.argmax(-1)
        action = jnp.where(trial < epsilon, random_action, greedy_action)
        return next_obs, action, q_values, key

    # put data in the last index
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    envs.async_reset()

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rb_add_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    storage = []
    one_step_storage = []

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree_map(lambda *xs: jnp.stack(xs), *storage)

    for update in range(1, args.num_updates + 2):
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        d2h_time = 0
        env_send_time = 0
        num_steps_with_bootstrap = args.num_steps + args.boostrap_n + int(len(storage) == 0)
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if args.concurrency:
            if update != 2:
                params = params_queue.get()
                # NOTE: block here is important because otherwise this thread will call
                # the jitted `get_action` function that hangs until the params are ready.
                # This blocks the `get_action` function in other actor threads.
                # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                params.network_params["params"]["Dense_0"][
                    "kernel"
                ].block_until_ready()  # TODO: check if params.block_until_ready() is enough
                actor_policy_version += 1
        else:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        rollout_time_start = time.time()
        for _ in range(args.train_frequency if update != 0 else args.train_frequency+1): # required to populate one_step_storage
            for _ in range(
                args.async_update, (num_steps_with_bootstrap) * args.async_update
            ):  # num_steps + 1 to get the states for value bootstrapping.
                env_recv_time_start = time.time()
                next_obs, next_reward, next_done, info = envs.recv()
                env_recv_time += time.time() - env_recv_time_start
                global_step += len(next_done) * args.num_actor_threads * len_actor_device_ids * args.world_size
                env_id = info["env_id"]

                inference_time_start = time.time()
                epsilon = epsilon_linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
                next_obs, next_action, next_q_value, key = get_action(params, next_obs, key)
                inference_time += time.time() - inference_time_start

                d2h_time_start = time.time()
                cpu_action = np.array(action)
                d2h_time += time.time() - d2h_time_start
                env_send_time_start = time.time()
                envs.send(cpu_action, env_id)
                env_send_time += time.time() - env_send_time_start
                storage_time_start = time.time()

                # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
                # so we use our own truncated flag
                truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
                if len(one_step_storage) != 0:
                    obs, action, q_value, last_env_id = one_step_storage.pop(0)
                    assert (last_env_id == env_id).all(), "last_env_id and env_id must be aligned in order for obs and next_obs to be aligned"
                    storage.append(
                        Transition(
                            obs=obs,
                            next_obs=next_obs,
                            dones=next_done,
                            actions=action,
                            rewards=next_reward,
                            truncations=truncated,
                            terminations=info["terminated"],
                            firststeps=info["elapsed_step"] == 0,
                        )
                    )
                episode_returns[env_id] += info["reward"]
                returned_episode_returns[env_id] = np.where(
                    info["terminated"] + truncated, episode_returns[env_id], returned_episode_returns[env_id]
                )
                episode_returns[env_id] *= (1 - info["terminated"]) * (1 - truncated)
                episode_lengths[env_id] += 1
                returned_episode_lengths[env_id] = np.where(
                    info["terminated"] + truncated, episode_lengths[env_id], returned_episode_lengths[env_id]
                )
                episode_lengths[env_id] *= (1 - info["terminated"]) * (1 - truncated)
                
                # CRITICAL: easy to over look
                one_step_storage.append(next_obs, next_action, next_q_value, env_id)
                
                storage_time += time.time() - storage_time_start
            rollout_time.append(time.time() - rollout_time_start)
            
        rb_add_time_start = time.time()
        rollout = prepare_data(storage)
        priorities = jnp.ones((args.num_envs * args.train_frequency, args.num_steps))
        rb.add(priorities, rollout)
        rb_add_time.append(time.time()- rb_add_time_start)

        # move bootstrapping step to the beginning of the next update
        storage = storage[-args.async_update :]
        
        avg_episodic_return = np.mean(returned_episode_returns)
        payload = (
            global_step,
            actor_policy_version,
            update,
            np.mean(params_queue_get_time),
            device_thread_id,
        )
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)


        if update % args.log_frequency == 0:
            if device_thread_id == 0:
                print(
                    f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}"
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
            writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)
            writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
            writer.add_scalar("stats/inference_time", inference_time, global_step)
            writer.add_scalar("stats/storage_time", storage_time, global_step)
            writer.add_scalar("stats/d2h_time", d2h_time, global_step)
            writer.add_scalar("stats/env_send_time", env_send_time, global_step)
            writer.add_scalar("stats/rollout_queue_put_time", np.mean(rollout_queue_put_time), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar(
                "charts/SPS_update",
                int(
                    args.local_num_envs
                    * args.num_steps
                    * len_actor_device_ids
                    * args.num_actor_threads
                    * args.world_size
                    / (time.time() - update_time_start)
                ),
                global_step,
            )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.async_batch_size = args.local_num_envs  # local_num_envs must be equal to async_batch_size due to limitation of `rlax`
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.async_update = int(args.local_num_envs / args.async_batch_size)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    print("global_learner_decices", global_learner_decices)
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"
    if args.track and args.local_rank == 0:
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

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    learner_keys = jax.device_put_replicated(key, learner_devices)

    # env setup
    envs = make_env(args.env_id, args.seed, args.local_num_envs, args.async_batch_size)()

    def lr_linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    network = DuelingQNetwork(action_dim=envs.single_action_space.n)
    network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    agent_state = TrainState.create(
        apply_fn=None,
        params=network_params,
        tx=optax.chain(
            # optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=lr_linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
    print(network.tabulate(network_key, np.array([envs.single_observation_space.sample()])))

    @jax.jit
    def get_q_value(
        params: flax.core.FrozenDict,
        obs: np.ndarray,
    ):
        return DuelingQNetwork(envs.single_action_space.n).apply(params, obs)

    @jax.jit
    def single_device_update(
        agent_state: TrainState,
        batch: Batch,
        key: jax.random.PRNGKey,
    ):
        discounts = (1 - batch.dones) * args.gamma
        next_q_values = get_q_value(agent_state.target, batch.next_obs)
        
        @partial(jax.value_and_grad, has_aux=True)
        def dqn_loss(params):
            q_values = get_q_value(params, batch.next_obs)
            td_error = jax.vmap(rlax.q_learning)(
                q_tm1=q_values,
                a_tm1=batch.actions,
                r_t=batch.rewards,
                discount_t=discounts,
                q_t=next_q_values
            )
            weighted_loss = td_error # * batch.weights
            return rlax.l2_loss(weighted_loss).mean(), (q_values, jnp.abs(td_error))
            
        (loss, (q_values, priorities)), grads = dqn_loss(agent_state.params)
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, loss, q_values, priorities, key

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
    )

    rollout_queue = queue.Queue(maxsize=args.num_actor_threads)
    params_queues = []
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None
    
    rb = ReplayBuffer(
        args.buffer_capacity,
        args.num_steps,
        args.boostrap_n,
        args.num_envs,
        args.seed,
    )

    unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
            params_queue = queue.Queue(maxsize=1)
            params_queue.put(device_params)
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(key, local_devices[d_id]),
                    args,
                    rollout_queue,
                    params_queue,
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    learner_devices,
                    d_idx * args.num_actor_threads + thread_id,
                    local_devices[d_id],
                    rb,
                ),
            ).start()
            params_queues.append(params_queue)

    @jax.jit
    def split_data(data):
        return jax.tree_map(lambda *xs: jnp.split(xs, len(learner_devices)), data)
    
    def put_params_to_actors(agent_state):
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_network_version = 0
    while True:
        rollout_queue_get_time_start = time.time()
        for _ in range(args.num_actor_threads * len(args.actor_device_ids)):
            (
                global_step,
                actor_policy_version,
                update,
                avg_params_queue_get_time,
                device_thread_id,
            ) = rollout_queue.get()
        if not rb.ready(args.learning_starts):
            put_params_to_actors(agent_state)
            continue
        batch, inds = rb.sample(args.local_batch_size)
        partitioned_batch = split_data(batch)
        sharded_batch = Batch(
            *list(map(lambda x: jax.device_put_sharded(x, devices=learner_devices), partitioned_batch))
        )
        learner_network_version += 1
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        (agent_state, loss, q_values, updated_priorities, learner_keys) = multi_device_update(
            agent_state,
            sharded_batch,
            learner_keys,
        )
        
        if learner_network_version % args.target_update_frequency == 0:
            agent_state.replace(target=agent_state.params)
        
        put_params_to_actors(agent_state)

        # record rewards for plotting purposes
        if learner_network_version % args.log_frequency == 0:
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                global_step,
            )
            writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
            writer.add_scalar("stats/rollout_queue_size", rollout_queue.qsize(), global_step)
            writer.add_scalar("stats/params_queue_size", params_queue.qsize(), global_step)
            print(
                global_step,
                f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_network_version={learner_network_version}, training time: {time.time() - training_time_start}s",
            )
            writer.add_scalar(
                "charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"][0].item(), global_step
            )
            writer.add_scalar("losses/value_loss", q_values[-1, -1, -1].item(), global_step)
            writer.add_scalar("losses/loss", loss[-1, -1, -1].item(), global_step)

        if update >= args.num_updates:
            break

    if args.save_model and args.local_rank == 0:
        if args.distributed:
            jax.distributed.shutdown()
        agent_state = flax.jax_utils.unreplicate(agent_state)
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params.network_params,
                            agent_state.params.actor_params,
                            agent_state.params.critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_envpool_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
                extra_dependencies=["jax", "envpool", "atari"],
            )

    envs.close()
    writer.close()