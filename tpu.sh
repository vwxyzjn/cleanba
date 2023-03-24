# set up TPU vms (both TPUv3 and TPUv4)
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

sudo apt update; sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

pyenv install 3.9.5
pyenv global 3.9.5
git clone https://github.com/vwxyzjn/cleanba.git
cd cleanba
pip install poetry
poetry install
poetry run pip install "jax[tpu]==0.3.25" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --actor-device-ids 0 --learner-device-ids 1 --local-num-envs 120 --async-batch-size 60 --track

# Run TPUv3 experiments
export WANDB_ENTITY=openrlbenchmark
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2+3+4+5+6_d1_tpuv3_8 --learner-device-ids 1 2 3 4 5 6 --local-num-envs 120 --async-batch-size 40 --wandb-project-name cleanba --track" \
    --num-seeds 3 \
    --workers 1


# Create preemptible TPUv4
gcloud alpha compute tpus queued-resources create your-queued-resource-id \
--node-id your-node-id \
--project cleanrl \
--zone us-central2-b \
--accelerator-type v4-8 \
--runtime-version tpu-vm-tf-2.11.0 \
--best-effort

gcloud alpha compute tpus queued-resources describe your-queued-resource-id \
--project cleanrl \
--zone us-central2-b \


# Run TPUv4 experiments
export WANDB_ENTITY=openrlbenchmark
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2_d1_tpuv4_8 --learner-device-ids 1 2 --local-num-envs 120 --async-batch-size 40 --wandb-project-name cleanba --track" \
    --num-seeds 3 \
    --workers 1

export WANDB_ENTITY=openrlbenchmark
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1_d1_tpuv4_8 --learner-device-ids 1 --local-num-envs 120 --async-batch-size 40 --wandb-project-name cleanba --track" \
    --num-seeds 3 \
    --workers 1


# Run `cleanba_ppo_envpool_impala_atari_wrapper_a0_l1_d2_tpuv4_8` TPUv4 experiments
export WANDB_TAGS=v0.0.1-16-g32dbf31
TPU_CHIPS_PER_PROCESS_BOUNDS=1,2,1 \
  TPU_PROCESS_BOUNDS=2,1,1 \
  TPU_PROCESS_ADDRESSES=localhost:8479,localhost:8480 \
  TPU_VISIBLE_DEVICES=0,1 \
  TPU_PROCESS_PORT=8479 \
  SLURM_JOB_ID=26017 SLURM_STEP_NODELIST=localhost SLURM_NTASKS=2 SLURM_PROCID=0 SLURM_LOCALID=0 SLURM_STEP_NUM_NODES=2 \
  CLOUD_TPU_TASK_ID=0 python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --distributed --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1_d2_tpuv4_8 --learner-device-ids 1 --wandb-project-name cleanba --track --seed 1

TPU_CHIPS_PER_PROCESS_BOUNDS=1,2,1 \
  TPU_PROCESS_BOUNDS=2,1,1 \
  TPU_PROCESS_ADDRESSES=localhost:8479,localhost:8480 \
  TPU_VISIBLE_DEVICES=2,3 \
  TPU_PROCESS_PORT=8480 \
  SLURM_JOB_ID=26017 SLURM_STEP_NODELIST=localhost SLURM_NTASKS=2 SLURM_PROCID=1 SLURM_LOCALID=0 SLURM_STEP_NUM_NODES=2 \
  CLOUD_TPU_TASK_ID=1 python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --distributed --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1_d2_tpuv4_8 --learner-device-ids 1 --seed 1
