# cleanba

Cleanba is CleanRL's implementation of DeepMind's Sebulba distributed training framework.

>**Note** that this is a WIP made public because it's easier for me to share points to . We are still working on the documentation and the codebase.


```
poetry install
poetry run pip install --upgrade "jax[cuda]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


## Usage

```
poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py
poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --help
```

Let us use `a0-l1,2,3-d1` to denote our setups, where `a0` means actor on GPU 0, `l1,2,3` means learner on GPUs 1,2,3, and `d1` means the computation is distributed 1 time.
Here are come common setups:

```
# a0-l0-d1: single GPU
python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --actor-device-ids 0 --learner-device-ids 0 --track
# a0-l0,1-d1: two GPUs
python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --actor-device-ids 0 --learner-device-ids 0 1 --track
# a0-l1,2-d1: three GPUs
python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --actor-device-ids 0 --learner-device-ids 1 2 --track
# a0-l1,2,3-d1: four GPUs
python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --actor-device-ids 0 --learner-device-ids 1 2 3 --track
# a0-l1,2,3,4-d1: five GPUs
python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --actor-device-ids 0 --learner-device-ids 1 2 3 4 --track
# a0-l1,2,3,4,5,6-d1: seven GPUs
python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --actor-device-ids 0 --learner-device-ids 1 2 3 4 5 6 --track

# a0-l0-d2: 8 GPUs (distributed 2 times on 4 GPUs)
# execute them in separate terminals; here we assume all 8 GPUs are on the same machine
# however it is possible to scale to hundreds of GPUs allowed by `jax.distributed`
CUDA_VISIBLE_DEVICES="0,1,2,3" SLURM_JOB_ID=26017 SLURM_STEP_NODELIST=localhost SLURM_NTASKS=2 SLURM_PROCID=0 SLURM_LOCALID=0 SLURM_STEP_NUM_NODES=2 python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --distributed --actor-device-ids 0 --learner-device-ids 1 2 3
CUDA_VISIBLE_DEVICES="4,5,6,7" SLURM_JOB_ID=26017 SLURM_STEP_NODELIST=localhost SLURM_NTASKS=2 SLURM_PROCID=1 SLURM_LOCALID=0 SLURM_STEP_NUM_NODES=2 python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --distributed --actor-device-ids 0 --learner-device-ids 1 2 3
```

