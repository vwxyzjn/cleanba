# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_atari_jaxpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import deque
import psutil 
from typing import NamedTuple
import threading
import queue
from functools import partial

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import envpool
import gym
import numpy as np

import distrax
import rlax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flax.training.train_state import TrainState
# from stable_baselines3.common.buffers import ReplayBuffer
from tensorboardX import SummaryWriter

f32 = np.float32

class Transition(NamedTuple):
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

class Batch(NamedTuple):
    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    # weights: np.ndarray

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
        shapes = {k: v.shape[2:] for k, v in example._asdict().items()} #[2:] removes time and batch dimensions
        # self._check_memory(example)

        self.total_sampleable_indicies = self.capacity * self.valid_length
        self.priorities = np.zeros((self.capacity, self.valid_length), dtype=f32)
        self.obs = np.zeros((self.capacity, self.length, *shapes['obs']), dtype=f32)
        self.rewards = np.zeros((self.capacity, self.length), dtype=f32)
        self.actions = np.zeros((self.capacity, self.length, *shapes['actions']), dtype=f32)
        self.dones = np.zeros((self.capacity, self.length), dtype=f32)

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
        self.dones[indexer] = np.array(rollout.dones).copy()

        self.timesteps_seen += self.n_envs * self.valid_length
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
            inds = np.random.randint(1, self.pos*self.valid_length, size=batch_size)
        weights = np.ones_like(inds, dtype=f32)
        return self._get_samples(inds, weights), inds

    def _priority_sampling(self, batch_size: int):
        N = self.capacity * self.valid_length # (number of rollouts * indexable length)
        
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
        )
    
    def update_priorities(self, inds: np.ndarray, updated_priorities: np.ndarray):
        b, t = np.unravel_index(inds, self.priorities.shape)
        self.priorities[b, t] = updated_priorities

    def size(self):
        return self.pos if not self.full else self.capacity

    def ready(self, min_size: int = None):
        return self.timesteps_seen > min_size or self.full

    # def _check_memory(self, rollout: NamedTuple):
    #     shapes = {k: v.shape[1:] for k, v in rollout._asdict().items()} # remove n_envs
    #     rollout_bytes = sum(np.prod(shape) for shape in shapes.values())
    #     total_bytes = self.capacity  * rollout_bytes
    #     if psutil is not None:
    #         avail_bytes = psutil.virtual_memory().available
    #     if total_bytes > avail_bytes:
    #         avail_bytes /= 1024 ** 3
    #         total_bytes /= 1024 ** 3
    #         warnings.warn(
    #             """This system does not have enough memory to store the replay buffer.
    #             Available memory: {avail_bytes:.2f} GB
    #             Required memory: {total_bytes:.2f} GB
    #             Difference: {avail_bytes - total_bytes:.2f} GB"""
    #         )

class ReplayBuffer(PERReplayBuffer):
    def __init__(self, capacity: int, length: int, valid_length: int, n_envs: int, seed: int = 0, alpha: float = 1, beta: float = 1) -> None:
        super().__init__(capacity, length, valid_length, n_envs, seed, alpha, beta)
        self.lock = threading.Lock()

    def thread_safe_add(self, priorities, rollout):
        with self.lock:
            self.add(priorities, rollout)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Apex DQN",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Breakout-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=25,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--rollout-length", type=int, default=2,
        help="number of environment steps before sending data to replay memory")
    parser.add_argument("--bootstrap-n", type=int, default=1,
        help="number of steps to bootstrap Q-value")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")

    args = parser.parse_args()

    return args

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

def make_env(env_id, num_envs, seed):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = args.num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        return envs
    return thunk


def scale_gradient(x: jnp.ndarray, scale: float) -> jnp.ndarray:
    """Multiplies the gradient of `x` by `scale`."""
    @jax.custom_gradient
    def wrapped(x: jnp.ndarray):
        return x, lambda grad: (grad * scale,)
    return wrapped(x)

scale = 1 / np.sqrt(2)

class QNetwork(nn.Module):
    action_dim: int

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
        
        x = scale_gradient(x, scale)

        # value stream
        v = nn.Dense(512)(x)
        v = nn.relu(v)
        value = nn.Dense(1)(v)
        
        # adv stream
        a = nn.Dense(512)(x)
        a = nn.relu(a)
        adv = nn.Dense(self.action_dim)(a)

        q_values = value + (adv - jnp.mean(adv, axis=1, keepdims=True)) # TODO: possible tthis mean isnt over the rigth direction
        return q_values


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def epilson_values_fn(N, epsilon=0.4, alpha=7.):
    i_range = jnp.arange(N)
    epsilon_values = epsilon ** (1 + (i_range / (N - 1)) * alpha)
    return jnp.expand_dims(epsilon_values, axis=-1)


if __name__ == "__main__":
    args = parse_args()
    args.total_train_steps = 4_000_000
    local_devices = jax.devices()
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

    # env setup
    envs = make_env(args.env_id, args.num_envs, args.seed)()

    obs = envs.reset()

    q_network = QNetwork(action_dim=envs.single_action_space.n)

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    rb = ReplayBuffer(
        args.buffer_size,
        args.rollout_length,
        args.bootstrap_n,
        args.num_envs,
        args.seed
    )

    ####################################################################
    ######################### START ACTOR ##############################

    def actor(
        rb,
        q_state_queue,
        ratio_deque,
    ):
        envs = make_env(args.env_id, args.num_envs, args.seed)()
        envs = RecordEpisodeStatistics(envs)
        epsilons = epilson_values_fn(args.num_envs)

        @jax.jit
        def get_action(params, obs, key):
            q_values = q_network.apply(params, obs)
            action = distrax.EpsilonGreedy(q_values, epsilons).sample(seed=key)
            return q_values, action

        rng = jax.random.PRNGKey(args.seed)
        avg_returns = deque(maxlen=20)
        obs = envs.reset()
        storage = []
        actor_start_time = time.time()
        actor_network_version = 0 
        for step in range(args.total_timesteps // args.num_envs):
            rng, key = jax.random.split(rng)
            global_step = step * args.num_envs
            
            if q_state_queue.qsize() != 0:
                q_state = q_state_queue.get()
                actor_network_version += 1
            
            q_values, actions = get_action(q_state.params, obs, key)
            actions = jax.device_get(actions)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            idx = -1 # greediest actor
            if done[idx] and info["lives"][idx] == 0: # only plot the first vectorized env
                ratio_deque.append(global_step / (time.time() - actor_start_time))
                print(f"global_step={global_step}, episodic_return={info['r'][idx]}. network_version: {actor_network_version}")
                avg_returns.append(info["r"][idx])
                writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

            transition = Transition(obs, actions, reward, done)
            storage.append(transition)
            if len(storage) == args.rollout_length:
                rollout = jax.tree_map(lambda *xs: np.asarray(xs).swapaxes(0,1), *storage)
                priorities = np.ones((args.num_envs, rb.n))
                rb.thread_safe_add(priorities, rollout)
                storage = storage[-1:]

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

    ######################### END ACTOR ##############################
    ##################################################################

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        discounts = (1 - dones) * args.gamma
        q_next_selector = q_network.apply(q_state.params, next_observations)  # (batch_size, num_actions)
        q_next_value = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            td_error = jax.vmap(rlax.double_q_learning)(
                q_tm1=q_pred,
                a_tm1=actions.flatten().astype(jnp.int32),
                r_t=rewards,
                discount_t=discounts,
                q_t_value=q_next_value,
                q_t_selector=q_next_selector
            )
            return (td_error ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    q_state_queue = queue.Queue(maxsize=1)
    q_state_queue.put(q_state)
    actor_sps_deque = deque(maxlen=10)
    threading.Thread(
        target=actor,
        args=(rb, q_state_queue, actor_sps_deque)
    ).start()

    while not rb.ready(args.learning_starts):
        time.sleep(1)

    def dataset_loader(rb, queue):
        while True:
            data, idxs = rb.sample(args.batch_size)
            queue.put((data, idxs))

    batch_queue = queue.Queue(16)
    args.num_dataloader_workers = 4
    for worker_id in range(args.num_dataloader_workers):
        threading.Thread(target=dataset_loader, args=(rb, batch_queue)).start()

    learner_sps_deque = deque(maxlen=10)
    train_step = 0
    while True:
        train_step += 1
        data, idxs = batch_queue.get()
        loss, old_val, q_state = update(
            q_state,
            data.obs,
            data.actions,
            data.next_obs,
            data.rewards,
            data.dones,
        )
        if train_step == 1: 
            # prevents including time to jit compile
            start_time = time.time()

        q_state_queue.put(q_state)

        # update target network
        if train_step % args.target_network_frequency == 0:
            q_state = q_state.replace(
                target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
            )

        # manage data generation:train ratio
        learner_sps_deque.append(train_step / (time.time() - start_time))
        actor_sps = (sum(actor_sps_deque) / len(actor_sps_deque)) * args.num_envs
        learner_sps = sum(learner_sps_deque) / len(learner_sps_deque)
        ratio = actor_sps / learner_sps

        if train_step % 100 == 0:
            writer.add_scalar("losses/td_loss", jax.device_get(loss), train_step)
            writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), train_step)
            print(f"SPS: actor {actor_sps:.1f} learner: {learner_sps:.1f} ratio: {ratio:.1f}")
            writer.add_scalar("charts/actor_SPS", actor_sps, train_step)
            writer.add_scalar("charts/learner_SPS", learner_sps, train_step)
            writer.add_scalar("charts/actor_learner_ratio", ratio, train_step)
            writer.add_scalar("charts/batch_queue_size", batch_queue.qsize(), train_step)
            writer.add_scalar("charts/train_step", train_step, train_step)


    ######################### END LEARNER ############################
    ####################################################################


    envs.close()
    writer.close()
