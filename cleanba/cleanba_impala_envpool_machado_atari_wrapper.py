import argparse
import os
import random
import time
import uuid
from collections import deque
from distutils.util import strtobool
from functools import partial
from typing import Sequence

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
import queue
import threading

import envpool
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
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
    parser.add_argument("--total-timesteps", type=int, default=50000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--local-num-envs", type=int, default=60,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=20,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--num-minibatches", type=int, default=2,
        help="the number of mini-batches")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--actor-device-ids", type=int, nargs="+", default=[0], # type is actually List[int]
        help="the device ids that actor workers will use (currently only support 1 device)")
    parser.add_argument("--learner-device-ids", type=int, nargs="+", default=[0], # type is actually List[int]
        help="the device ids that learner workers will use")
    parser.add_argument("--distributed", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use `jax.distirbuted`")
    parser.add_argument("--profile", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to call block_until_ready() for profiling")
    parser.add_argument("--test-actor-learner-throughput", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to test actor-learner throughput by removing the actor-learner communication")
    args = parser.parse_args()
    args.async_batch_size = args.local_num_envs # local_num_envs must be equal to async_batch_size due to limitation of `rlax`
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.local_batch_size
    args.async_update = int(args.local_num_envs / args.async_batch_size)
    assert len(args.actor_device_ids) == 1, "only 1 actor_device_ids is supported now"
    # fmt: on
    return args


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


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class Network(nn.Module):
    channelss: Sequence[int] = (16, 32, 32)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channelss:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@partial(jax.jit, static_argnums=(3))
def get_action(
    params: flax.core.FrozenDict,
    next_obs: np.ndarray,
    key: jax.random.PRNGKey,
    action_dim: int,
):
    next_obs = jnp.array(next_obs)
    hidden = Network().apply(params.network_params, next_obs)
    logits = Actor(action_dim).apply(params.actor_params, hidden)
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    return next_obs, action, logits, key


def prepare_data(
    obs: list,
    dones: list,
    actions: list,
    logitss: list,
    firststeps: list,
    env_ids: list,
    rewards: list,
):
    obs = jnp.asarray(obs)
    dones = jnp.asarray(dones)
    actions = jnp.asarray(actions)
    logitss = jnp.asarray(logitss)
    firststeps = jnp.asarray(firststeps)
    env_ids = jnp.asarray(env_ids)
    rewards = jnp.asarray(rewards)
    return obs, dones, actions, logitss, firststeps, env_ids, rewards


@jax.jit
def make_bulk_array(
    obs: list,
    actions: list,
    logitss: list,
):
    obs = jnp.asarray(obs)
    actions = jnp.asarray(actions)
    logitss = jnp.asarray(logitss)
    return obs, actions, logitss


def rollout(
    key: jax.random.PRNGKey,
    args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
):
    envs = make_env(args.env_id, args.seed + jax.process_index(), args.local_num_envs, args.async_batch_size)()
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()

    # put data in the last index
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    envs.async_reset()

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    obs = []
    dones = []
    actions = []
    logitss = []
    env_ids = []
    rewards = []
    truncations = []
    terminations = []
    firststeps = [] # first step of an episode
    for update in range(1, args.num_updates + 2):
        # NOTE: This is a major difference from the sync version:
        # at the end of the rollout phase, the sync version will have the next observation
        # ready for the value bootstrap, but the async version will not have it.
        # for this reason we do `num_steps + 1`` to get the extra states for value bootstrapping.
        # but note that the extra states are not used for the loss computation in the next iteration,
        # while the sync version will use the extra state for the loss computation.
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        env_send_time = 0

        num_steps_with_bootstrap = args.num_steps + 1 + int(len(obs)==0)
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if update != 2:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)
        rollout_time_start = time.time()
        for _ in range(
            args.async_update, (num_steps_with_bootstrap) * args.async_update
        ):  # num_steps + 1 to get the states for value bootstrapping.
            env_recv_time_start = time.time()
            next_obs, next_reward, next_done, info = envs.recv()
            env_recv_time += time.time() - env_recv_time_start
            global_step += len(next_done) * len_actor_device_ids * args.world_size
            env_id = info["env_id"]

            inference_time_start = time.time()
            next_obs, action, logits, key = get_action(params, next_obs, key, envs.single_action_space.n)
            inference_time += time.time() - inference_time_start

            env_send_time_start = time.time()
            envs.send(np.array(action), env_id)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()
            obs.append(next_obs)
            dones.append(next_done)
            actions.append(action)
            logitss.append(logits)
            env_ids.append(env_id)
            rewards.append(next_reward)
            firststeps.append(info["elapsed_step"] == 0)

            # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
            # so we use our own truncated flag
            truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
            truncations.append(truncated)
            terminations.append(info["terminated"])
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
            storage_time += time.time() - storage_time_start
        if args.profile:
            action.block_until_ready()
        rollout_time.append(time.time() - rollout_time_start)
        writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)

        avg_episodic_return = np.mean(returned_episode_returns)
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        writer.add_scalar("stats/truncations", np.sum(truncations), global_step)
        writer.add_scalar("stats/terminations", np.sum(terminations), global_step)
        writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
        writer.add_scalar("stats/inference_time", inference_time, global_step)
        writer.add_scalar("stats/storage_time", storage_time, global_step)
        writer.add_scalar("stats/env_send_time", env_send_time, global_step)
        # `make_bulk_array` is actually important. It accumulates the data from the lists
        # into single bulk arrays, which later makes transferring the data to the learner's
        # device slightly faster. See https://wandb.ai/costa-huang/cleanRL/reports/data-transfer-optimization--VmlldzozNjU5MTg1
        c_obs, c_actions, c_logitss = obs, actions, logitss
        if args.learner_device_ids[0] != args.actor_device_ids[0]:
            c_obs, c_actions, c_logitss = make_bulk_array(
                obs,
                actions,
                logitss,
            )

        payload = (
            global_step,
            actor_policy_version,
            update,
            c_obs,
            c_actions,
            c_logitss,
            firststeps,
            dones,
            env_ids,
            rewards,
            np.mean(params_queue_get_time),
        )
        if update == 1 or not args.test_actor_learner_throughput:
            rollout_queue_put_time_start = time.time()
            rollout_queue.put(payload)
            rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)
            writer.add_scalar("stats/rollout_queue_put_time", np.mean(rollout_queue_put_time), global_step)

        writer.add_scalar(
            "charts/SPS_update",
            int(
                args.local_num_envs
                * args.num_steps
                * len_actor_device_ids
                * args.world_size
                / (time.time() - update_time_start)
            ),
            global_step,
        )

        obs = obs[-args.async_update:]
        dones = dones[-args.async_update:]
        actions = actions[-args.async_update:]
        logitss = logitss[-args.async_update:]
        env_ids = env_ids[-args.async_update:]
        rewards = rewards[-args.async_update:]
        truncations = truncations[-args.async_update:]
        terminations = terminations[-args.async_update:]
        firststeps = firststeps[-args.async_update:]


@partial(jax.jit, static_argnums=(2))
def get_action_and_value2(
    params: flax.core.FrozenDict,
    x: np.ndarray,
    action_dim: int,
):
    hidden = Network().apply(params.network_params, x)
    raw_logits = Actor(action_dim).apply(params.actor_params, hidden)
    value = Critic().apply(params.critic_params, hidden).squeeze()
    return raw_logits, value


def policy_gradient_loss(logits, *args):
  """rlax.policy_gradient_loss, but with sum(loss) and [T, B, ...] inputs."""
  mean_per_batch = jax.vmap(rlax.policy_gradient_loss, in_axes=1)(logits, *args)
  total_loss_per_batch = mean_per_batch * logits.shape[0]
  return jnp.sum(total_loss_per_batch)

def entropy_loss_fn(logits, *args):
  """rlax.entropy_loss, but with sum(loss) and [T, B, ...] inputs."""
  mean_per_batch = jax.vmap(rlax.entropy_loss, in_axes=1)(logits, *args)
  total_loss_per_batch = mean_per_batch * logits.shape[0]
  return jnp.sum(total_loss_per_batch)


def impala_loss(params, x, a, logitss, rewards, dones, firststeps, action_dim):
    discounts = (1.0 - dones) * args.gamma
    mask = (1.0 - firststeps)
    policy_logits, newvalue = jax.vmap(get_action_and_value2, in_axes=(None, 0, None))(params, x, action_dim)

    v_t = newvalue[1:]
    # Remove bootstrap timestep from non-timesteps.
    v_tm1 = newvalue[:-1]
    policy_logits = policy_logits[:-1]
    logitss = logitss[:-1]
    a = a[:-1]
    mask = mask[:-1]
    rewards = rewards[:-1]
    discounts = discounts[:-1]

    rhos = rlax.categorical_importance_sampling_ratios(policy_logits, logitss, a)
    vtrace_td_error_and_advantage = jax.vmap(
        rlax.vtrace_td_error_and_advantage, in_axes=1, out_axes=1)

    vtrace_returns = vtrace_td_error_and_advantage(
        v_tm1, v_t, rewards, discounts, rhos)
    pg_advs = vtrace_returns.pg_advantage
    pg_loss = policy_gradient_loss(policy_logits, a, pg_advs, mask)

    baseline_loss = 0.5 * jnp.sum(jnp.square(vtrace_returns.errors) * mask)
    ent_loss = entropy_loss_fn(policy_logits, mask)

    total_loss = pg_loss
    total_loss += args.vf_coef * baseline_loss
    total_loss += args.ent_coef * ent_loss
    return total_loss, (pg_loss, baseline_loss, ent_loss)


@partial(jax.jit, static_argnames=("action_dim"))
def single_device_update(
    agent_state: TrainState,
    obs,
    actions,
    logitss,
    rewards,
    dones,
    firststeps,
    action_dim,
    key: jax.random.PRNGKey,
):
    impala_loss_grad_fn = jax.value_and_grad(impala_loss, has_aux=True)
    def update_minibatch(agent_state, minibatch):
        mb_obs, mb_actions, mb_logitss, mb_rewards, mb_dones, mb_firststeps = minibatch
        (loss, (pg_loss, v_loss, entropy_loss)), grads = impala_loss_grad_fn(
            agent_state.params,
            mb_obs,
            mb_actions,
            mb_logitss,
            mb_rewards,
            mb_dones,
            mb_firststeps,
            action_dim,
        )
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, (loss, pg_loss, v_loss, entropy_loss)

    agent_state, (loss, pg_loss, v_loss, entropy_loss) = jax.lax.scan(
        update_minibatch,
        agent_state,
        (
            jnp.array(jnp.split(obs, args.num_minibatches, axis=1)),
            jnp.array(jnp.split(actions, args.num_minibatches, axis=1)),
            jnp.array(jnp.split(logitss, args.num_minibatches, axis=1)),
            jnp.array(jnp.split(rewards, args.num_minibatches, axis=1)),
            jnp.array(jnp.split(dones, args.num_minibatches, axis=1)),
            jnp.array(jnp.split(firststeps, args.num_minibatches, axis=1)),
        ),
    )
    return agent_state, loss, pg_loss, v_loss, entropy_loss, key


if __name__ == "__main__":
    args = parse_args()
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    args.async_update = int(args.local_num_envs / args.async_batch_size)
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
    envs = make_env(args.env_id, args.seed, args.local_num_envs, args.async_batch_size)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches)) / args.num_updates
        return args.learning_rate * frac

    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params,
            actor.init(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
            critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
        out_axes=(0, 0, 0, 0, 0, None),
        static_broadcasted_argnums=(7),
    )

    rollout_queue = queue.Queue(maxsize=1)
    params_queues = []
    for d_idx, d_id in enumerate(args.actor_device_ids):
        params_queue = queue.Queue(maxsize=1)
        params_queue.put(jax.device_put(flax.jax_utils.unreplicate(agent_state.params), local_devices[d_id]))
        threading.Thread(
            target=rollout,
            args=(
                jax.device_put(key, local_devices[d_id]),
                args,
                rollout_queue,
                params_queue,
                writer,
                learner_devices,
            ),
        ).start()
        params_queues.append(params_queue)

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    prepare_data = jax.jit(prepare_data, device=learner_devices[0])
    while True:
        learner_policy_version += 1
        if learner_policy_version == 1 or not args.test_actor_learner_throughput:
            rollout_queue_get_time_start = time.time()
            (
                global_step,
                actor_policy_version,
                update,
                obs,
                actions,
                logitss,
                firststeps,
                dones,
                env_ids,
                rewards,
                avg_params_queue_get_time,
            ) = rollout_queue.get()
            rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
            writer.add_scalar("stats/rollout_params_queue_get_time_diff", np.mean(rollout_queue_get_time) - avg_params_queue_get_time, global_step)


        data_transfer_time_start = time.time()
        obs, dones, actions, logitss, firststeps, env_ids, rewards = prepare_data(
            obs,
            dones,
            actions,
            logitss,
            firststeps,
            env_ids,
            rewards,
        )
        
        obs = jnp.array_split(obs, len(learner_devices), axis=1)
        actions = jnp.array_split(actions, len(learner_devices), axis=1)
        logitss = jnp.array_split(logitss, len(learner_devices), axis=1)
        rewards = jnp.array_split(rewards, len(learner_devices), axis=1)
        dones = jnp.array_split(dones, len(learner_devices), axis=1)
        firststeps = jnp.array_split(firststeps, len(learner_devices), axis=1)
        data_transfer_time.append(time.time() - data_transfer_time_start)
        writer.add_scalar("stats/data_transfer_time", np.mean(data_transfer_time), global_step)

        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, entropy_loss, key) = multi_device_update(
            agent_state,
            jax.device_put_sharded(obs, learner_devices),
            jax.device_put_sharded(actions, learner_devices),
            jax.device_put_sharded(logitss, learner_devices),
            jax.device_put_sharded(rewards, learner_devices),
            jax.device_put_sharded(dones, learner_devices),
            jax.device_put_sharded(firststeps, learner_devices),
            envs.single_action_space.n,
            key,
        )
        if learner_policy_version == 1 or not args.test_actor_learner_throughput:
            for d_idx, d_id in enumerate(args.actor_device_ids):
                params_queues[d_idx].put(jax.device_put(flax.jax_utils.unreplicate(agent_state.params), local_devices[d_id]))
        if args.profile:
            v_loss[-1, -1, -1].block_until_ready()
        writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
        writer.add_scalar("stats/rollout_queue_size", rollout_queue.qsize(), global_step)
        writer.add_scalar("stats/params_queue_size", params_queue.qsize(), global_step)
        print(
            global_step,
            f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"][0].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)
        if update >= args.num_updates:
            break

        # print weights
        # sum_params(agent_state.params)
        # print("network_params", agent_state.params.network_params['params']["Dense_0"]["kernel"])
        # print("actor_params", agent_state.params.actor_params['params']["Dense_0"]["kernel"])
        # print("critic_params", agent_state.params.critic_params['params']["Dense_0"]["kernel"])

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
