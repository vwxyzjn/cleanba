import argparse
import collections
import os
import random
import time
import uuid
from collections import deque
from distutils.util import strtobool
from functools import partial
from types import SimpleNamespace
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
    parser.add_argument("--local-num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=20,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--num-minibatches", type=int, default=1,
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
        help="the device ids that actor workers will use")
    parser.add_argument("--num-actor-threads", type=int, default=2,
        help="the number of actor threads to use (currently only support 1 thread)")
    parser.add_argument("--learner-device-ids", type=int, nargs="+", default=[0], # type is actually List[int]
        help="the device ids that learner workers will use")
    parser.add_argument("--distributed", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use `jax.distirbuted`")
    args = parser.parse_args()
    args.async_batch_size = args.local_num_envs # local_num_envs must be equal to async_batch_size due to limitation of `rlax`
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    # args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    args.async_update = int(args.local_num_envs / args.async_batch_size)
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


@partial(jax.jit, static_argnames=("action_dim"))
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


def rollout(
    agent_state_store,
    key: jax.random.PRNGKey,
    args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    actor_device,
):
    envs = make_env(
        args.env_id,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs,
        args.async_batch_size,
    )()
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
    time.sleep(20 + (0.1382 / (len_actor_device_ids * args.num_actor_threads)) * device_thread_id)

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
    firststeps = []  # first step of an episode

    def prepare_data(
        obs: list,
        dones: list,
        actions: list,
        logitss: list,
        firststeps: list,
        env_ids: list,
        rewards: list,
    ):
        obs = jnp.array_split(jnp.asarray(obs), len(learner_devices), axis=1)
        dones = jnp.array_split(jnp.asarray(dones), len(learner_devices), axis=1)
        actions = jnp.array_split(jnp.asarray(actions), len(learner_devices), axis=1)
        logitss = jnp.array_split(jnp.asarray(logitss), len(learner_devices), axis=1)
        firststeps = jnp.array_split(jnp.asarray(firststeps), len(learner_devices), axis=1)
        env_ids = jnp.array_split(jnp.asarray(env_ids), len(learner_devices), axis=1)
        rewards = jnp.array_split(jnp.asarray(rewards), len(learner_devices), axis=1)
        return obs, dones, actions, logitss, firststeps, env_ids, rewards

    prepare_data = jax.jit(prepare_data, device=actor_device)
    for update in range(1, args.num_updates + 2):
        if update == 4 and device_thread_id == 0:
            jax.profiler.start_trace(f"runs/{run_name}/profile", create_perfetto_trace=True)

        # print("update", update, "agent_state", agent_state_store[0].step)
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

        num_steps_with_bootstrap = args.num_steps + 1 + int(len(obs) == 0)
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if update != 2:
            # params = params_queue.get()
            # agent_state_store[0].step
            local_agent_state = agent_state_store[0]
            params = jax.device_put(flax.jax_utils.unreplicate(local_agent_state.params), actor_device)
            actor_policy_version = local_agent_state.step
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)
        rollout_time_start = time.time()
        for step in range(
            args.async_update, (num_steps_with_bootstrap) * args.async_update
        ):  # num_steps + 1 to get the states for value bootstrapping.

            # if step % 4 == 0:
            #     local_agent_state = agent_state_store[0]
            #     params = jax.device_put(flax.jax_utils.unreplicate(local_agent_state.params), actor_device)
            #     actor_policy_version = local_agent_state.step

            env_recv_time_start = time.time()
            next_obs, next_reward, next_done, info = envs.recv()
            env_recv_time += time.time() - env_recv_time_start
            global_step += len(next_done) * args.num_actor_threads * len_actor_device_ids * args.world_size
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
        rollout_time.append(time.time() - rollout_time_start)
        writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)

        avg_episodic_return = np.mean(returned_episode_returns)
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
        if device_thread_id == 0:
            print(
                f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}"
            )
            print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        writer.add_scalar("stats/truncations", np.sum(truncations), global_step)
        writer.add_scalar("stats/terminations", np.sum(terminations), global_step)
        writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
        writer.add_scalar("stats/inference_time", inference_time, global_step)
        writer.add_scalar("stats/storage_time", storage_time, global_step)
        writer.add_scalar("stats/env_send_time", env_send_time, global_step)
        c_obs, c_actions, c_logitss = obs, actions, logitss
        p_obs, p_dones, p_actions, p_logitss, p_firststeps, p_env_ids, p_rewards = prepare_data(
            obs,
            dones,
            actions,
            logitss,
            firststeps,
            env_ids,
            rewards,
        )
        payload = (
            global_step,
            actor_policy_version,
            update,
            jax.device_put_sharded(p_obs, devices=learner_devices),
            jax.device_put_sharded(p_dones, devices=learner_devices),
            jax.device_put_sharded(p_actions, devices=learner_devices),
            jax.device_put_sharded(p_logitss, devices=learner_devices),
            jax.device_put_sharded(p_firststeps, devices=learner_devices),
            jax.device_put_sharded(p_env_ids, devices=learner_devices),
            jax.device_put_sharded(p_rewards, devices=learner_devices),
            np.mean(params_queue_get_time),
            device_thread_id,
        )
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
                * args.num_actor_threads
                * args.world_size
                / (time.time() - update_time_start)
            ),
            global_step,
        )

        obs = obs[-args.async_update :]
        dones = dones[-args.async_update :]
        actions = actions[-args.async_update :]
        logitss = logitss[-args.async_update :]
        env_ids = env_ids[-args.async_update :]
        rewards = rewards[-args.async_update :]
        truncations = truncations[-args.async_update :]
        terminations = terminations[-args.async_update :]
        firststeps = firststeps[-args.async_update :]
        print("update", update)
        if update == 15:
            if device_thread_id == 0:
                jax.profiler.stop_trace()
            return


@partial(jax.jit, static_argnames=("action_dim"))
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
    mask = 1.0 - firststeps
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
    vtrace_td_error_and_advantage = jax.vmap(rlax.vtrace_td_error_and_advantage, in_axes=1, out_axes=1)

    vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, rewards, discounts, rhos)
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
    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_gradient_updates = args.total_timesteps // args.local_batch_size
    args.num_updates = args.total_timesteps // int(
        args.local_batch_size * args.num_actor_threads * len(args.actor_device_ids) * args.world_size
    )
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
        frac = 1.0 - (count // (args.num_minibatches)) / args.num_gradient_updates
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
    agent_state_store = collections.deque(maxlen=1)
    agent_state_store.append(agent_state)

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
        out_axes=(0, 0, 0, 0, 0, None),
        static_broadcasted_argnums=(7),
    )

    rollout_queue = queue.Queue(maxsize=args.num_actor_threads * len(args.actor_device_ids))
    params_queues = []

    import multiprocessing as mp

    num_cpus = mp.cpu_count()
    fair_num_cpus = num_cpus // len(args.actor_device_ids)
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(flax.jax_utils.unreplicate(agent_state.params), local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
            threading.Thread(
                target=rollout,
                args=(
                    agent_state_store,
                    jax.device_put(key, local_devices[d_id]),
                    args,
                    rollout_queue,
                    None,
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    learner_devices,
                    d_idx * args.num_actor_threads + thread_id,
                    local_devices[d_id],
                ),
            ).start()

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    while True:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        (
            global_step,
            actor_policy_version,
            update,
            t_obs,
            t_dones,
            t_actions,
            t_logitss,
            t_firststeps,
            t_env_ids,
            t_rewards,
            avg_params_queue_get_time,
            device_thread_id,
        ) = rollout_queue.get()
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
        writer.add_scalar(
            "stats/rollout_params_queue_get_time_diff",
            np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
            global_step,
        )
        data_transfer_time_start = time.time()

        writer.add_scalar(
            f"stats/actor_policy_version_lag/{device_thread_id}",
            learner_policy_version * args.num_minibatches - actor_policy_version[0].item(),
        )
        data_transfer_time.append(time.time() - data_transfer_time_start)
        writer.add_scalar("stats/data_transfer_time", np.mean(data_transfer_time), global_step)

        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, entropy_loss, key) = multi_device_update(
            agent_state,
            t_obs,
            t_actions,
            t_logitss,
            t_rewards,
            t_dones,
            t_firststeps,
            envs.single_action_space.n,
            key,
        )
        agent_state_store.append(agent_state)
        # for d_idx, d_id in enumerate(args.actor_device_ids):
        #     device_params = jax.device_put(flax.jax_utils.unreplicate(agent_state.params), local_devices[d_id])
        #     for thread_id in range(args.num_actor_threads):
        #         params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

        writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
        writer.add_scalar("stats/rollout_queue_size", rollout_queue.qsize(), global_step)
        # writer.add_scalar("stats/params_queue_size", params_queue.qsize(), global_step)
        print(
            global_step,
            f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version * args.num_minibatches}, training time: {time.time() - training_time_start}s",
        )
        writer.add_scalar("stats/learner_policy_version", learner_policy_version * args.num_minibatches, global_step)

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

        if update == 15:
            break

    envs.close()
    writer.close()

    # recursively search for the path of `perfetto_trace.json.gz` in `f"runs/{run_name}/profile"` and upload it to wandb
    HTML_TEMPLATE = r"""
<div>
run the following to enable reverse proxy:
</div>
<div>
mitmproxy -p 27658 --mode reverse:https://api.wandb.ai --modify-headers '/~s/Access-Control-Allow-Origin/*'
</div>
<form action="https://codepen.io/pen/define" method="POST" target="_blank">
  <input type="hidden" name="data" value='{"title": "New Pen!", "html": "
  <!doctype html>
  <html lang=\"en-us\">
  <link rel=\"shortcut icon\" href=\"data:image/x-icon;,\" type=\"image/x-icon\">
  <body>

  <style>
    pre {
      border: 1px solid #eee;
      margin: 10px 0;
      font-family: monospace;
      font-size: 10px;
      min-height: 100px;
    }
    
    body > * { margin: 20px; }
    
    #btn_fetch {  font-size: 14px; }
  </style>

  <div>
    <select id=\"source\" size=2>
    <option selected>{perfetto_trace_json_gz_path}</option>
    </select>

    <br>
    <button type=\"button\" id=\"btn_fetch\">Fetch and open trace</button>
    <br>
    <pre id=\"logs\" cols=\"80\" rows=\"20\"></pre>
  </div>
</body>
<script type=\"text/javascript\">
  const ORIGIN = \"https://ui.perfetto.dev\";
  
  const logs = document.getElementById(\"logs\");
  const btnFetch = document.getElementById(\"btn_fetch\");
  
  async function fetchAndOpen(traceUrl) {
    logs.innerText += `Fetching trace from ${traceUrl}...\\n`;
    const resp = await fetch(traceUrl);
    // Error checcking is left as an exercise to the reader.
    const blob = await resp.blob();
    const arrayBuffer = await blob.arrayBuffer();
    logs.innerText += `fetch() complete, now passing to ui.perfetto.dev\\n`;
    openTrace(arrayBuffer, traceUrl);
  }
  function openTrace(arrayBuffer, traceUrl) {
    const win = window.open(ORIGIN);
    if (!win) {
      btnFetch.style.background = \"#f3ca63\";
      btnFetch.onclick = () => openTrace(arrayBuffer);
      logs.innerText += `Popups blocked, you need to manually click the button`;
      btnFetch.innerText = \"Popups blocked, click here to open the trace file\";
      return;
    }
    const timer = setInterval(() => win.postMessage(\"PING\", ORIGIN), 50);
    const onMessageHandler = (evt) => {
      if (evt.data !== \"PONG\") return;
  
      // We got a PONG, the UI is ready.
      window.clearInterval(timer);
      window.removeEventListener(\"message\", onMessageHandler);
  
      const reopenUrl = new URL(location.href);
      reopenUrl.hash = `#reopen=${traceUrl}`;
      win.postMessage({
        perfetto: {
          buffer: arrayBuffer,
          title: \"The Trace Title\",
          url: reopenUrl.toString(),
      }}, ORIGIN);
    };

    window.addEventListener(\"message\", onMessageHandler);
  }

  if (location.hash.startsWith(\"#reopen=\")) {
   const traceUrl = location.hash.substr(8);
   fetchAndOpen(traceUrl);
  }
  btnFetch.onclick = () => fetchAndOpen(document.getElementById(\"source\").value);
  </script>
</body>
</html>
"}'>

  <input type="submit" value="Create New Pen with Prefilled Data">
</form>
    """

    HTML_TEMPLATE2 = r"""
<!doctype html>
<html lang="en-us">
<link rel="shortcut icon" href="data:image/x-icon;," type="image/x-icon">
<body>

Please open a new tab and open

{content}
    """
    time.sleep(20)

    def find_perfetto_trace_json_gz(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file == "perfetto_trace.json.gz":
                    return os.path.join(root, file)
        return None

    perfetto_trace_json_gz_path = find_perfetto_trace_json_gz(f"runs/{run_name}/profile")
    if perfetto_trace_json_gz_path is not None and args.track:
        wandb.save(perfetto_trace_json_gz_path, base_path=perfetto_trace_json_gz_path[: -len("perfetto_trace.json.gz")])
        with open(f"runs/{run_name}/perfetto.html", "w") as f:
            f.write(
                HTML_TEMPLATE.replace(
                    r"{perfetto_trace_json_gz_path}", f"https://localhost:27658/files/{wandb.run.path}/perfetto_trace.json.gz"
                )
            )
        wandb.save(f"runs/{run_name}/perfetto.html", base_path=f"runs/{run_name}/")
        wandb.log(
            {
                "perfetto_trace": wandb.Html(
                    HTML_TEMPLATE2.replace(r"{content}", f"https://api.wandb.ai/files/{wandb.run.path}/perfetto.html")
                )
            }
        )
