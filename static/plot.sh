# 10 CPU setting
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=torchbeast&ceik=env&cen=exp_name&metric=mean_episode_return' \
        'monobeast_cpu10?cl=Monobeast IMPALA, 1 A100, 10 CPU' \
    --filters '?we=costa-huang&wpn=moolib-atari-2&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado?cl=Moolib IMPALA, 1 A100, 10 CPU' \
    --filters '?we=openrlbenchmark&wpn=cleanba&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'cleanba_impala_a0_l0_d1_cpu10?tag=v1.0.0b2&cl=Cleanba IMPALA, 1 A100, 10 CPU' \
        'cleanba_ppo_sync_a0_l0_d1_cpu10?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO (Sync), 1 A100, 10 CPU' \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 8.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 50000 \
    --rc.performance_profile_num_bootstrap_reps 50000 \
    --rc.interval_estimates_num_bootstrap_reps 50000 \
    --output-filename cleanba/main_10CPU \
    --scan-history  --offline

# 40 CPU setting
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=torchbeast&ceik=env&cen=exp_name&metric=mean_episode_return' \
        'monobeast_cpu80?cl=Monobeast IMPALA' \
    --filters '?we=openrlbenchmark&wpn=moolib-atari&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado_40cpu?cl=Moolib (Resnet CNN) 1 A100, 40 CPU'  \
    --filters '?we=openrlbenchmark&wpn=cleanba&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'cleanba_impala_a0_l0_d1_cpu40?tag=v1.0.0b3-2-g36b430e&cl=Cleanba IMPALA' \
        'cleanba_ppo_sync_a0_l0_d1_cpu40?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO' \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 8.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 50000 \
    --rc.performance_profile_num_bootstrap_reps 50000 \
    --rc.interval_estimates_num_bootstrap_reps 50000 \
    --output-filename cleanba/main_40CPU \
    --scan-history  --offline

# spec out setting
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=torchbeast&ceik=env&cen=exp_name&metric=mean_episode_return' \
        'monobeast_cpu80?cl=Monobeast IMPALA, 1 A100, 80 CPU' \
    --filters '?we=openrlbenchmark&wpn=moolib-atari&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado_8gpu_actor_batch_size16?cl=Moolib IMPALA, 8 A100, 80 CPU'  \
    --filters '?we=openrlbenchmark&wpn=cleanba&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'cleanba_impala_a0_l1_d4_cpu46?tag=v1.0.0b2&cl=Cleanba IMPALA, 8 A100, 46 CPU' \
        'cleanba_ppo_sync_a0_l0_d8_cpu46?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO (Sync), 8 A100, 46 CPU' \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 8.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 50000 \
    --rc.performance_profile_num_bootstrap_reps 50000 \
    --rc.interval_estimates_num_bootstrap_reps 50000 \
    --output-filename cleanba/spec_out \
    --scan-history  --offline


# smootheness 
python -m openrlbenchmark.rlops \
    --filters '?we=costa-huang&wpn=moolib-atari-2&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado?cl=Moolib (Resnet CNN) 1 A100, 10 CPU' \
    --filters '?we=openrlbenchmark&wpn=moolib-atari&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado_8gpu_actor_batch_size16?cl=Moolib (Resnet CNN) 8 A100, 80 CPU'  \
    --filters '?we=openrlbenchmark&wpn=cleanba&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'cleanba_impala_a0_l0_d1_cpu10?tag=v1.0.0b2&cl=Cleanba IMPALA, 1 A100, 10 CPU' \
        'cleanba_impala_a0_l1_d4_cpu46?tag=v1.0.0b2&cl=Cleanba IMPALA, 8 A100, 46 CPU' \
        'cleanba_ppo_sync_a0_l0_d1_cpu10?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO (Sync), 1 A100, 10 CPU' \
        'cleanba_ppo_sync_a0_l0_d8_cpu46?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO (Sync), 8 A100, 46 CPU' \
    --env-ids Assault-v5 Asterix-v5 Breakout-v5 Boxing-v5 ChopperCommand-v5 DemonAttack-v5 Gravitar-v5 Kangaroo-v5 KungFuMaster-v5 NameThisGame-v5 TimePilot-v5 UpNDown-v5   \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --pc.cm 2.3 \
    --pc.rm 1.8 \
    --pc.hspace 0.6 \
    --pc.wspace 0.4 \
    --output-filename cleanba/smootheness \
    --scan-history --offline

python -m openrlbenchmark.rlops \
    --filters '?we=costa-huang&wpn=moolib-atari-2&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado?cl=Moolib (Resnet CNN) 1 A100, 10 CPU' \
    --filters '?we=openrlbenchmark&wpn=moolib-atari&ceik=env_id&cen=exp_name&metric=global/mean_episode_return' \
        'moolib_impala_envpool_machado_8gpu_actor_batch_size16?cl=Moolib (Resnet CNN) 8 A100, 80 CPU'  \
    --filters '?we=openrlbenchmark&wpn=cleanba&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'cleanba_impala_a0_l0_d1_cpu10?tag=v1.0.0b2&cl=Cleanba IMPALA, 1 A100, 10 CPU' \
        'cleanba_impala_a0_l1_d4_cpu46?tag=v1.0.0b2&cl=Cleanba IMPALA, 8 A100, 46 CPU' \
        'cleanba_ppo_sync_a0_l0_d1_cpu10?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO (Sync), 1 A100, 10 CPU' \
        'cleanba_ppo_sync_a0_l0_d8_cpu46?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO (Sync), 8 A100, 46 CPU' \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5  \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --pc.cm 2.3 \
    --pc.rm 1.8 \
    --pc.hspace 0.6 \
    --pc.wspace 0.4 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 8.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 50000 \
    --rc.performance_profile_num_bootstrap_reps 50000 \
    --rc.interval_estimates_num_bootstrap_reps 50000 \
    --output-filename cleanba/smootheness_complete \
    --scan-history --offline


# cleanrl
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanba&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'ppo_atari_envpool_xla_jax_scan?tag=v0.0.1-31-gb5e05f8&cl=CleanRL PPO (Sync), 1 A100, 10 CPU' \
        'cleanba_ppo_sync_a0_l0_d8_cpu46?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO (Sync), 8 A100, 46 CPU' \
        'cleanba_ppo_a0+1_l2+3_d2_cpu46?tag=v1.0.0b3&cl=Cleanba PPO, 8 A100, 46 CPU' \
        'cleanba_impala_sync_a0_l1_d4_cpu46?tag=v1.0.0b3-3-g81dca00&cl=Cleanba IMPALA (Sync), 8 A100, 46 CPU' \
        'cleanba_impala_a0_l1_d4_cpu46?tag=v1.0.0b2&cl=Cleanba IMPALA, 8 A100, 46 CPU' \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 8.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 50000 \
    --rc.performance_profile_num_bootstrap_reps 50000 \
    --rc.interval_estimates_num_bootstrap_reps 50000 \
    --output-filename cleanba/cleanrl \
    --scan-history  --offline


# direct comparison between IMPALA and PPO
python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanba&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'cleanba_impala_a0_l0_d1_cpu10?tag=v1.0.0b2&cl=Cleanba IMPALA, 1 A100, 10 CPU' \
        'cleanba_ppo_fair_a0_l0_d1_cpu10?tag=v1.0.0b3-3-g81dca00&cl=Cleanba PPO (Fair), 1 A100, 10 CPU' \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 8.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 50000 \
    --rc.performance_profile_num_bootstrap_reps 50000 \
    --rc.interval_estimates_num_bootstrap_reps 50000 \
    --output-filename cleanba/ppo_vs_impala_fair \
    --scan-history 

