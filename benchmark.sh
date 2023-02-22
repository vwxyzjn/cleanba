export WANDB_ENTITY=openrlbenchmark
python -m cleanrl_utils.benchmark \
    --env-ids  BigfishHard-v0 BossfightHard-v0 CaveflyerHard-v0 ChaserHard-v0 ClimberHard-v0 CoinrunHard-v0 DodgeballHard-v0 FruitbotHard-v0 HeistHard-v0 JumperHard-v0 LeaperHard-v0 MazeHard-v0 MinerHard-v0 NinjaHard-v0 PlunderHard-v0 StarpilotHard-v0 \
    --command "poetry run python cleanba/cleanba_ppo_envpool_procgen.py --distributed --track --save-model --upload-model" \
    --num-seeds 1 \
    --workers 10 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 4 \
    --slurm-nodes 1 \
    --slurm-template-path cleanba.slurm_template
