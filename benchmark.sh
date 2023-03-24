export WANDB_ENTITY=openrlbenchmark

# prod buffering ablation
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/ppo_envpool_impala_atari_wrapper.py --track --wandb-project-name cleanba" \
    --num-seeds 1 \
    --start-seed 3 \
    --workers 1 \
    --slurm-gpus-per-task 1 \
    --slurm-total-cpus 50 \
    --slurm-ntasks 1 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/ppo_atari_envpool_xla_jax_scan.py --track --wandb-project-name cleanba" \
    --num-seeds 1 \
    --start-seed 2 \
    --workers 1 \
    --slurm-gpus-per-task 1 \
    --slurm-total-cpus 50 \
    --slurm-ntasks 1 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l0_d1_b120 --actor-device-ids 0 --learner-device-ids 0 --local-num-envs 120 --async-batch-size 120 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 1 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l0_d1_b60 --actor-device-ids 0 --learner-device-ids 0 --local-num-envs 120 --async-batch-size 60 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 1 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l0_d1_b40 --actor-device-ids 0 --learner-device-ids 0 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 1 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l0_d1_b40_no_concurrency --concurrency False --actor-device-ids 0 --learner-device-ids 0 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 1 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

# prod procgen
python -m cleanrl_utils.benchmark \
    --env-ids  BigfishHard-v0 BossfightHard-v0 CaveflyerHard-v0 ChaserHard-v0 ClimberHard-v0 CoinrunHard-v0 DodgeballHard-v0 FruitbotHard-v0 HeistHard-v0 JumperHard-v0 LeaperHard-v0 MazeHard-v0 MinerHard-v0 NinjaHard-v0 PlunderHard-v0 StarpilotHard-v0 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_procgen.py --distributed --track --wandb-project-name cleanba --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 1 \
    --workers 16 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 4 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

# prod cleanba_ppo_envpool_impala_atari_wrapper_naturecnn
python -m cleanrl_utils.benchmark \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper_naturecnn.py --distributed --learner-device-ids 1 --track --wandb-project-name cleanba --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 3 \
    --workers 35 \
    --slurm-gpus-per-task 2 \
    --slurm-ntasks 2 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

# prod cleanba_ppo_envpool_impala_atari_wrapper
python -m cleanrl_utils.benchmark \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --distributed --learner-device-ids 1 2 3 --track --wandb-project-name cleanba --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 3 \
    --workers 57 \
    --slurm-gpus-per-task 4 \
    --slurm-ntasks 2 \
    --slurm-nodes 1 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

# prod cleanba_ppo_envpool_machado_atari_wrapper
python -m cleanrl_utils.benchmark \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_machado_atari_wrapper.py --distributed --learner-device-ids 1 2 3 --track --wandb-project-name cleanba --save-model --upload-model --hf-entity cleanrl" \
    --num-seeds 2 \
    --start-seed 2 \
    --workers 57 \
    --slurm-gpus-per-task 4 \
    --slurm-ntasks 2 \
    --slurm-nodes 1 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

# prod reproducibility 
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l0_d1 --actor-device-ids 0 --learner-device-ids 0 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 1 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l0_d2 --distributed --actor-device-ids 0 --learner-device-ids 0  --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 2 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1_d1 --actor-device-ids 0 --learner-device-ids 1 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 2 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l0+1_d1 --actor-device-ids 0 --learner-device-ids 0 1 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 2 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2_d1 --actor-device-ids 0 --learner-device-ids 1 2 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 3 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2+3_d1 --actor-device-ids 0 --learner-device-ids 1 2 3 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 4 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2+3+4_d1 --actor-device-ids 0 --learner-device-ids 1 2 3 4 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 5 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2+3+4+5+6_d1 --actor-device-ids 0 --learner-device-ids 1 2 3 4 5 6 --local-num-envs 120 --async-batch-size 40 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 7 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l0_d4 --distributed --actor-device-ids 0 --learner-device-ids 0 --local-num-envs 30 --async-batch-size 30 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 3 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 4 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template

# large batch size
python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2+3_d4 --distributed --total-timesteps 100000000 --anneal-lr False --learner-device-ids 1 2 3 --track --wandb-project-name cleanba" \
    --num-seeds 1 \
    --workers 3 \
    --slurm-gpus-per-task 4 \
    --slurm-ntasks 4 \
    --slurm-total-cpus 120 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2+3_d8 --distributed --total-timesteps 100000000 --anneal-lr False --learner-device-ids 1 2 3 --track --wandb-project-name cleanba" \
    --num-seeds 1 \
    --workers 3 \
    --slurm-gpus-per-task 4 \
    --slurm-ntasks 8 \
    --slurm-total-cpus 240 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2+3_d16 --distributed --total-timesteps 100000000 --anneal-lr False --learner-device-ids 1 2 3 --track --wandb-project-name cleanba" \
    --num-seeds 1 \
    --workers 3 \
    --slurm-gpus-per-task 4 \
    --slurm-ntasks 16 \
    --slurm-total-cpus 480 \
    --slurm-template-path cleanba.slurm_template

python -m cleanrl_utils.benchmark \
    --env-ids Breakout-v5 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_impala_atari_wrapper.py --exp-name cleanba_ppo_envpool_impala_atari_wrapper_a0_l1+2+3_d32 --distributed --total-timesteps 100000000 --anneal-lr False --learner-device-ids 1 2 3 --track --wandb-project-name cleanba" \
    --num-seeds 1 \
    --workers 3 \
    --slurm-gpus-per-task 4 \
    --slurm-ntasks 32 \
    --slurm-total-cpus 960 \
    --slurm-template-path cleanba.slurm_template

# cleanba_impala_envpool_machado_atari_wrapper
python -m cleanrl_utils.benchmark \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --command "poetry run python cleanba/cleanba_impala_envpool_machado_atari_wrapper.py --distributed --learner-device-ids 1 2 3 --track --wandb-project-name cleanba" \
    --num-seeds 3 \
    --workers 57 \
    --slurm-gpus-per-task 4 \
    --slurm-ntasks 2 \
    --slurm-total-cpus 50 \
    --slurm-template-path cleanba.slurm_template


