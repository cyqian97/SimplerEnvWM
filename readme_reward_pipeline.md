# Vision Reward Pipeline


#### Running PPO with Vision Rewards

Train PPO using vision-based rewards (SAM3 + Depth-Anything-3):

```bash
./docker_train_ppo_rgb.sh
```

Optional arguments:
- `ENV_ID` (default: `PickCube-v1`)
- `NUM_ENVS` (default: `256`)
- `TOTAL_TIMESTEPS` (default: `10000000`)
- `RUNS_DIR` (default: `./runs`)

Example:
```bash
./docker_train_ppo_rgb.sh PickCube-v1 256 10000000 ./runs
```

