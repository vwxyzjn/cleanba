import os
import queue
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, NamedTuple, Optional, Sequence

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
from rich.pretty import pprint
from tensorboardX import SummaryWriter

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


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
    total_timesteps: int = 50000000
    "total timesteps of the experiments"
    learning_rate: float = 0.0006
    "the learning rate of the optimizer"
    local_num_envs: int = 64
    "the number of parallel game environments"
    num_actor_threads: int = 2
    "the number of actor threads to use"
    num_steps: int = 20
    "the number of steps to run in each environment per policy rollout"
    anneal_lr: bool = True
    "Toggle learning rate annealing for policy and value networks"
    gamma: float = 0.99
    "the discount factor gamma"
    num_minibatches: int = 4
    "the number of mini-batches"
    gradient_accumulation_steps: int = 1
    "the number of gradient accumulation steps before performing an optimization step"
    ent_coef: float = 0.01
    "coefficient of the entropy"
    vf_coef: float = 0.5
    "coefficient of the value function"
    max_grad_norm: float = 40.0
    "the maximum norm for the gradient clipping"
    channels: List[int] = field(default_factory=lambda: [16, 32, 32])
    "the channels of the CNN"
    hiddens: List[int] = field(default_factory=lambda: [256])
    "the hiddens size of the MLP"

    actor_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that actor workers will use"
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that learner workers will use"
    distributed: bool = False
    "whether to use `jax.distirbuted`"
    concurrency: bool = True
    "whether to run the actor and learner concurrently"

    # runtime arguments to be filled in
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    num_updates: int = 0
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


def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
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


### RMSProp implementation for PyTorch-style RMSProp
# see https://github.com/deepmind/optax/issues/532#issuecomment-1676371843
# fmt: off
import jax
import jax.numpy as jnp
from optax import update_moment_per_elem_norm
from optax._src import base, combine, transform
from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate
from optax._src.transform import ScaleByRmsState


def scale_by_rms_pytorch_style(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.
) -> base.GradientTransformation:
  """See https://github.com/deepmind/optax/issues/532#issuecomment-1676371843"""

  def init_fn(params):
    nu = jax.tree_util.tree_map(
        lambda n: jnp.full_like(n, initial_scale), params)  # second moment
    return ScaleByRmsState(nu=nu)

  def update_fn(updates, state, params=None):
    del params
    nu = update_moment_per_elem_norm(updates, state.nu, decay, 2)
    updates = jax.tree_util.tree_map(
        lambda g, n: g / (jax.lax.sqrt(n) + eps), updates, nu)
    return updates, ScaleByRmsState(nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def rmsprop_pytorch_style(
    learning_rate: ScalarOrSchedule,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.,
    momentum: Optional[float] = None,
    nesterov: bool = False
) -> base.GradientTransformation:
  return combine.chain(
      scale_by_rms_pytorch_style(
          decay=decay, eps=eps, initial_scale=initial_scale),
      _scale_by_learning_rate(learning_rate),
      (transform.trace(decay=momentum, nesterov=nesterov)
       if momentum is not None else base.identity())
  )
# fmt: on


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class Network(nn.Module):
    channels: Sequence[int] = (16, 32, 32)
    hiddens: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channels:
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


class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    logitss: list
    env_ids: list
    rewards: list
    truncations: list
    terminations: list
    firststeps: list  # first step of an episode


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
    envs = make_env(
        args.env_id,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs,
    )()
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
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
    storage = []

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree_map(lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage)

    for update in range(1, args.num_updates + 2):
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        d2h_time = 0
        env_send_time = 0
        num_steps_with_bootstrap = (
            args.num_steps + 1 + int(len(storage) == 0)
        )  # num_steps + 1 to get the states for value bootstrapping.
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
        for _ in range(1, num_steps_with_bootstrap):
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

            # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
            # so we use our own truncated flag
            truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
            storage.append(
                Transition(
                    obs=next_obs,
                    dones=next_done,
                    actions=action,
                    logitss=logits,
                    env_ids=env_id,
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
            storage_time += time.time() - storage_time_start
        rollout_time.append(time.time() - rollout_time_start)

        avg_episodic_return = np.mean(returned_episode_returns)
        partitioned_storage = prepare_data(storage)
        sharded_storage = Transition(
            *list(map(lambda x: jax.device_put_sharded(x, devices=learner_devices), partitioned_storage))
        )
        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            np.mean(params_queue_get_time),
            device_thread_id,
        )
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)

        # move bootstrapping step to the beginning of the next update
        storage = storage[-1:]

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
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
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
    pprint(args)

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
    envs = make_env(args.env_id, args.seed, args.local_num_envs)()

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches)) / args.num_updates
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
        tx=optax.MultiSteps(
            optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.inject_hyperparams(rmsprop_pytorch_style)(
                    learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=0.01, decay=0.99
                ),
            ),
            every_k_schedule=args.gradient_accumulation_steps,
        ),
    )
    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
    print(network.tabulate(network_key, np.array([envs.single_observation_space.sample()])))
    print(actor.tabulate(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))))
    print(critic.tabulate(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))))

    @jax.jit
    def get_logits_and_value(
        params: flax.core.FrozenDict,
        x: np.ndarray,
    ):
        hidden = Network(args.channels, args.hiddens).apply(params.network_params, x)
        raw_logits = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
        value = Critic().apply(params.critic_params, hidden).squeeze(-1)
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

    def impala_loss(params, x, a, logitss, rewards, dones, firststeps):
        discounts = (1.0 - dones) * args.gamma
        mask = 1.0 - firststeps
        policy_logits, newvalue = jax.vmap(get_logits_and_value, in_axes=(None, 0))(params, x)

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

    @jax.jit
    def single_device_update(
        agent_state: TrainState,
        sharded_storages: List[Transition],
        key: jax.random.PRNGKey,
    ):
        storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
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
            )
            grads = jax.lax.pmean(grads, axis_name="local_devices")
            agent_state = agent_state.apply_gradients(grads=grads)
            return agent_state, (loss, pg_loss, v_loss, entropy_loss)

        agent_state, (loss, pg_loss, v_loss, entropy_loss) = jax.lax.scan(
            update_minibatch,
            agent_state,
            (
                jnp.array(jnp.split(storage.obs, args.num_minibatches * args.gradient_accumulation_steps, axis=1)),
                jnp.array(jnp.split(storage.actions, args.num_minibatches * args.gradient_accumulation_steps, axis=1)),
                jnp.array(jnp.split(storage.logitss, args.num_minibatches * args.gradient_accumulation_steps, axis=1)),
                jnp.array(jnp.split(storage.rewards, args.num_minibatches * args.gradient_accumulation_steps, axis=1)),
                jnp.array(jnp.split(storage.dones, args.num_minibatches * args.gradient_accumulation_steps, axis=1)),
                jnp.array(jnp.split(storage.firststeps, args.num_minibatches * args.gradient_accumulation_steps, axis=1)),
            ),
        )
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
        v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
        entropy_loss = jax.lax.pmean(entropy_loss, axis_name="local_devices").mean()
        return agent_state, loss, pg_loss, v_loss, entropy_loss, key

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
    )

    params_queues = []
    rollout_queues = []
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            params_queues[-1].put(device_params)
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(key, local_devices[d_id]),
                    args,
                    rollout_queues[-1],
                    params_queues[-1],
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
        sharded_storages = []
        for d_idx, d_id in enumerate(args.actor_device_ids):
            for thread_id in range(args.num_actor_threads):
                (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    avg_params_queue_get_time,
                    device_thread_id,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_storages.append(sharded_storage)
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, entropy_loss, learner_keys) = multi_device_update(
            agent_state,
            sharded_storages,
            learner_keys,
        )
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

        # record rewards for plotting purposes
        if learner_policy_version % args.log_frequency == 0:
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                global_step,
            )
            writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
            writer.add_scalar("stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step)
            writer.add_scalar("stats/params_queue_size", params_queues[-1].qsize(), global_step)
            print(
                global_step,
                f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
            )
            writer.add_scalar(
                "charts/learning_rate", agent_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(), global_step
            )
            writer.add_scalar("losses/value_loss", v_loss[-1].item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1].item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss[-1].item(), global_step)
            writer.add_scalar("losses/loss", loss[-1].item(), global_step)
        if learner_policy_version >= args.num_updates:
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
                "IMPALA",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
                extra_dependencies=["jax", "envpool", "atari"],
            )

    envs.close()
    writer.close()
