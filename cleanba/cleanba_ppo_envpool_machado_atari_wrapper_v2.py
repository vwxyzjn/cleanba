import argparse
import os
import random
import time
import uuid
from collections import deque
from distutils.util import strtobool
from functools import partial
from typing import Sequence, Tuple, List, Optional
from types import SimpleNamespace
from dataclasses import dataclass, field

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
import tyro


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

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    "the id of the environment"
    total_timesteps: int = 50000000
    "total timesteps of the experiments"
    learning_rate: float = 2.5e-4
    "the learning rate of the optimizer"
    local_num_envs: int = 60
    "the number of parallel game environments"
    num_steps: int = 128
    "the number of steps to run in each environment per policy rollout"
    anneal_lr: bool = True
    "Toggle learning rate annealing for policy and value networks"
    gamma: float = 0.99
    "the discount factor gamma"
    gae_lambda: float = 0.95
    "the lambda for the general advantage estimation"
    num_minibatches: int = 4
    "the number of mini-batches"
    update_epochs: int = 4
    "the K epochs to update the policy"
    norm_adv: bool = True
    "Toggles advantages normalization"
    clip_coef: float = 0.1
    "the surrogate clipping coefficient"
    ent_coef: float = 0.01
    "coefficient of the entropy"
    vf_coef: float = 0.5
    "coefficient of the value function"
    max_grad_norm: float = 0.5
    "the maximum norm for the gradient clipping"
    target_kl: float = None
    "the target KL divergence threshold"
    channels: List[int] = field(default_factory=lambda: [64, 128, 128, 64])
    "the channels of the CNN"
    hiddens: List[int] = field(default_factory=lambda: [512, 512])
    "the hiddens size of the MLP"

    actor_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that actor workers will use"
    num_actor_threads: int = 2
    "the number of actor threads to use"
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that learner workers will use"
    distributed: bool = False
    "whether to use `jax.distirbuted`"

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
    minibatch_size: int = 0
    num_updates: int = 0
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
    channelss: Sequence[int] = (64, 128, 128, 64)
    hiddens: Sequence[int] = (512, 512)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channelss:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden in self.hiddens:
            x = nn.Dense(hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
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


def rollout(
    key: jax.random.PRNGKey,
    args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    actor_device,
):
    envs = make_env(args.env_id,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs,
        args.async_batch_size,
    )()
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()

    @jax.jit
    def get_action(
        params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        next_obs = jnp.array(next_obs)
        hidden = Network(args.channels, args.hiddens).apply(params.network_params, next_obs)
        logits = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return next_obs, action, logits, key
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
    firststeps = []  # first step of an episode
    @jax.jit
    def prepare_data(
        obs: list,
        dones: list,
        actions: list,
        logitss: list,
        firststeps: list,
        env_ids: list,
        rewards: list,
    ):
        obs = jnp.split(jnp.asarray(obs), len(learner_devices), axis=1)
        dones = jnp.split(jnp.asarray(dones), len(learner_devices), axis=1)
        actions = jnp.split(jnp.asarray(actions), len(learner_devices), axis=1)
        logitss = jnp.split(jnp.asarray(logitss), len(learner_devices), axis=1)
        firststeps = jnp.split(jnp.asarray(firststeps), len(learner_devices), axis=1)
        env_ids = jnp.split(jnp.asarray(env_ids), len(learner_devices), axis=1)
        rewards = jnp.split(jnp.asarray(rewards), len(learner_devices), axis=1)
        return obs, dones, actions, logitss, firststeps, env_ids, rewards
    for update in range(1, args.num_updates + 2):
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        d2h_time = 0
        env_send_time = 0

        num_steps_with_bootstrap = args.num_steps + 1 + int(len(obs) == 0)
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if update != 2:
            params = params_queue.get()
            # NOTE: block here is important because otherwise this thread will call
            # the jitted `get_action` function that hangs until the params are ready.
            # This blocks the `get_action` function in other actor threads.
            # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
            params.network_params['params']["Dense_0"]["kernel"].block_until_ready() # TODO: check if params.block_until_ready() is enough
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
            global_step += len(next_done) * args.num_actor_threads * len_actor_device_ids * args.world_size
            env_id = info["env_id"]

            inference_time_start = time.time()
            next_obs, action, logits, key = get_action(params, next_obs, key)
            inference_time += time.time() - inference_time_start

            d2h_time_start = time.time()
            cpu_action = np.array(action)
            d2h_time += time.time() - d2h_time_start
            env_send_time_start = time.time()
            envs.send(cpu_action, env_id)
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
            print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}")
            print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        writer.add_scalar("stats/truncations", np.sum(truncations), global_step)
        writer.add_scalar("stats/terminations", np.sum(terminations), global_step)
        writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
        writer.add_scalar("stats/inference_time", inference_time, global_step)
        writer.add_scalar("stats/storage_time", storage_time, global_step)
        writer.add_scalar("stats/d2h_time", d2h_time, global_step)
        writer.add_scalar("stats/env_send_time", env_send_time, global_step)
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

        obs = obs[-args.async_update:]
        dones = dones[-args.async_update:]
        actions = actions[-args.async_update:]
        logitss = logitss[-args.async_update:]
        env_ids = env_ids[-args.async_update:]
        rewards = rewards[-args.async_update:]
        truncations = truncations[-args.async_update:]
        terminations = terminations[-args.async_update:]
        firststeps = firststeps[-args.async_update:]


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.async_batch_size = args.local_num_envs # local_num_envs must be equal to async_batch_size due to limitation of `rlax`
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    args.async_update = int(args.local_num_envs / args.async_batch_size)
    assert args.local_num_envs % len(args.learner_device_ids) == 0, "local_num_envs must be divisible by len(learner_device_ids)"
    assert int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0, "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
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
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    network = Network(args.channels, args.hiddens)
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


    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        hidden = Network(args.channels, args.hiddens).apply(params.network_params, x)
        logits = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = Critic().apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value


    def ppo_loss(params, x, a, behavior_logprobs, rewards, dones, firststeps):
        discounts = (1.0 - dones) * args.gamma
        mask = 1.0 - firststeps
        newlogprob, entropy, newvalue = jax.vmap(get_action_and_value2, in_axes=(None, 0, 0))(params, x, a)

        behavior_logprobs = behavior_logprobs[:-1]
        newlogprob = newlogprob[:-1]
        entropy = entropy[:-1]
        a = a[:-1]
        mask = mask[:-1]
        rewards = rewards[:-1]
        discounts = discounts[:-1]

        def gae_advantages(rewards: jnp.array, discounts: jnp.array,
                        values: jnp.array) -> Tuple[jnp.ndarray, jnp.array]:
            advantages = rlax.truncated_generalized_advantage_estimation(rewards, discounts, args.gae_lambda, values)
            advantages = jax.lax.stop_gradient(advantages)
            target_values = values[:-1] + advantages
            target_values = jax.lax.stop_gradient(target_values)
            return advantages, target_values
        gae_advantages = jax.vmap(gae_advantages, in_axes=1, out_axes=1)
        advantages, target_values = gae_advantages(rewards, discounts, newvalue)

        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logratio = newlogprob - behavior_logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue[:-1] - target_values) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    @jax.jit
    def single_device_update(
        agent_state: TrainState,
        obs,
        actions,
        logitss,
        rewards,
        dones,
        firststeps,
        key: jax.random.PRNGKey,
    ):
        obs = jnp.hstack(obs)
        actions = jnp.hstack(actions)
        logitss = jnp.hstack(logitss)
        rewards = jnp.hstack(rewards)
        dones = jnp.hstack(dones)
        firststeps = jnp.hstack(firststeps)
        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        behavior_logprobs = jax.vmap(
            lambda logits, action: jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        )(logitss, actions)
        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def update_minibatch(agent_state, minibatch):
                mb_obs, mb_actions, mb_behavior_logprobs, mb_rewards, mb_dones, mb_firststeps = minibatch
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    mb_obs,
                    mb_actions,
                    mb_behavior_logprobs,
                    mb_rewards,
                    mb_dones,
                    mb_firststeps,
                )
                grads = jax.lax.pmean(grads, axis_name="local_devices")
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
                update_minibatch,
                agent_state,
                (
                    jnp.array(jnp.split(obs, args.num_minibatches, axis=1)),
                    jnp.array(jnp.split(actions, args.num_minibatches, axis=1)),
                    jnp.array(jnp.split(behavior_logprobs, args.num_minibatches, axis=1)),
                    jnp.array(jnp.split(rewards, args.num_minibatches, axis=1)),
                    jnp.array(jnp.split(dones, args.num_minibatches, axis=1)),
                    jnp.array(jnp.split(firststeps, args.num_minibatches, axis=1)),
                ),
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None),
        out_axes=(0, 0, 0, 0, 0, 0, None),
    )

    rollout_queue = queue.Queue(maxsize=args.num_actor_threads)
    params_queues = []

    import multiprocessing as mp
    num_cpus = mp.cpu_count()
    fair_num_cpus = num_cpus // len(args.actor_device_ids)
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x,y,z: None

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
                ),
            ).start()
            params_queues.append(params_queue)

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    while True:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        obs = []
        dones = []
        actions = []
        logitss = []
        firststeps = []
        env_ids = []
        rewards = []
        for _ in range(args.num_actor_threads * len(args.actor_device_ids)):
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
            obs.append(t_obs)
            dones.append(t_dones)
            actions.append(t_actions)
            logitss.append(t_logitss)
            firststeps.append(t_firststeps)
            env_ids.append(t_env_ids)
            rewards.append(t_rewards)
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
        writer.add_scalar("stats/rollout_params_queue_get_time_diff", np.mean(rollout_queue_get_time) - avg_params_queue_get_time, global_step)

        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key) = multi_device_update(
            agent_state,
            obs,
            actions,
            logitss,
            rewards,
            dones,
            firststeps,
            key,
        )
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

        writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
        writer.add_scalar("stats/rollout_queue_size", rollout_queue.qsize(), global_step)
        writer.add_scalar("stats/params_queue_size", params_queue.qsize(), global_step)
        print(
            global_step,
            f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if learner_policy_version % 50 == 0:
            writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"][0].item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss[-1, -1, -1].item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1, -1, -1].item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss[-1, -1, -1].item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl[-1, -1, -1].item(), global_step)
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
