# Approach-and-score football

`Mjlab-Football-Flat-MicroDuck` builds on the walking recipe with the
`groundcontact` robot and a 70 mm / 15 g ball. Its backlash twin is
`Mjlab-Football-Flat-Backlash-MicroDuck`, using the matching robot model.
The [README workflow](../README.md#football-approach-and-score) covers training,
resuming, visual playback, evaluation, and export.

## Task and scoring

The simulated ball-and-goal tracker supplies body-frame velocity commands.
The robot approaches a right-foot kicking stance, aligns toward the goal,
strikes, and receives exact-zero navigation commands afterward. Episodes last
eight seconds. This is a single-shot task, not dribbling or a match.

The goal has a 40 cm clear opening and 30 cm crossbar height, with solid posts
and rigid translucent panels representing a net. Its center is 75 cm from the
ball's initial position. The direction varies by approximately ±20 degrees
relative to the robot's initial heading.

A qualified strike requires right-foot/ball contact and left-foot/ground contact
in the same physics substep, plus standing and heading checks. Contact force
histories capture strikes shorter than a policy step. A score requires the
whole ball to cross the goal plane from the front, between the posts and below
the crossbar. Swept interpolation handles fast shots. A shot that crosses
outside the mouth cannot later score by entering from the side or recrossing.
A score is latched; success additionally requires standing at the end without
a balance failure. Later ball motion does not undo a goal.

Rewards pay only new per-episode records for approach, forward ball speed, and
forward distance, plus a one-time goal reward. Their weights are 20, 1, 3, and 8
respectively; speed and distance credit are capped at 1 m/s and 0.75 m.
A stationary or oscillating state cannot repeatedly collect progress rewards.
Overspeed is a nonnegative cost with negative weight. Inspect actual peak ball
speed separately: the reward cap is not a physical speed limit.

## Standing and exploration

Standing requires trunk height of at least 0.095 m above the ground, tilt below
30 degrees, and no non-foot body contact with the floor. Sustained failure for
0.12 seconds terminates an episode. Progress and strike credit require valid
posture immediately; balance failure remains latched until reset.

The height threshold was checked against the walking source: its fifth-percentile
trunk height was approximately 0.115 m. An angle-only check admitted a failed
policy with a level trunk only 0.057 m above the floor. The final recipe preserves
the walking leg-pose and head-tracking rewards and measures physical support
explicitly.

Exploration uses a task-specific Gaussian with standard deviation bounded to
0.05–0.30 rad; the source walking policy used approximately 0.16–0.24 rad.
This prevents runaway action noise. Deterministic actions and ONNX outputs
remain unfiltered. The learning rate is fixed at 1e-4 and entropy coefficient
is 0.001. The actor remains 61D and the critic adds ball state and task progress.

## Initialization, curricula, and resuming

Use a compatible walking checkpoint with the same observation semantics.
The training helper's `--from-walking` option loads the actor and its normalizer,
but starts a fresh critic, optimizer, iteration counter, and task curriculum.
It rejects a stand-expert-like twist normalizer and collapses inherited CoM
curricula to their mature ranges. Matching tensor dimensions alone does not
establish checkpoint compatibility.

`--from-policy` offers the same actor-only initialization from a compatible
football policy. `--resume` instead restores actor, critic, optimizer,
normalizers, and curriculum step, and starts at the next iteration after the
saved checkpoint. Its resume path uses the same mature CoM ranges, so use it
with football checkpoints from this walking-initialized recipe.
`--iterations` specifies additional iterations. A run resumed
from checkpoint 700 for 4,299 iterations ends at checkpoint 4999.

Initially, 25% of training spawns are beside the kicking foot; the others require
20–35 cm approaches with lateral variation. Maximum approach distance increases
to 60 cm at iteration 1000 and 90 cm at iteration 2000, with 20% near-ball starts.
Action-rate cost ramps from zero to -0.01 at iteration 500, -0.03 at 1500, and
-0.05 at 3000. Pushes begin at iteration 2000. These are initial schedule choices:
use evaluation results to decide whether a stage needs more time. The play
configuration uses approach starts out to 90 cm and no pushes.

The helper writes TensorBoard logs, checkpoints, environment/agent configurations,
and `initialization.json` under `logs/rsl_rl/football/<timestamp>_<run-name>/`.
Run long jobs inside a terminal multiplexer or a managed background process.
Training and playback require a CUDA GPU; CPU regression tests do not.

## Evaluation and inspection

`--eval-every 100` runs deterministic policy evaluations at saved checkpoints,
including the first save and final checkpoint. Each check uses 32 complete idle
and 32 complete approach episodes, with seed 123 and approach distances up to
90 cm. `latest_evaluation.json` records the checkpoint, status, and results.

By default training stops if more than 10% of episodes end in a fall/invalid
state, fewer than 90% finish standing, or fewer than 60% reach the kicking stance.
Pass `--continue-on-regression` to record failures as `regression_advisory` and
continue training. Physical terminations and scoring gates still apply.

For a larger standalone battery, `scripts/eval_football.py` defaults to 64
worlds and three episodes per world (192 episodes). `--idle` tests exact-zero
navigation commands, and `--near-probability 1` tests kicks from near-ball
starts. `--video <path.mp4>` records the first world’s first episode. The report
includes goals, qualified strikes, approach completion, standing/balance flags,
height/tilt quantiles, credited progress, and actual peak ball speed.

A 5,000-iteration policy scored and remained standing in 30/32 approach trials,
with all 32 reaching the stance, no falls, and 32/32 idle-standing passes.
Its local viewer rollout was also visually checked. These are simulation
results from the fixed-seed check, not a hardware or broad-generalization claim.
Weights are not included in this PR/repository; use your own training checkpoint.

If the local graphics driver crashes during video initialization, prepend
`LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` to the evaluation command.
This uses Mesa software rendering while simulation still runs on the GPU.

## Physics and regression checks

```bash
uv run scripts/check_football_physics.py
uv run --with pytest pytest tests/test_football.py tests/test_football_cfg.py
```

The physics checker holds calibrated BAM control targets for three seconds
from noisy starts and measures tilt, root height, ball drift, and initial
robot-ball penetration. It isolates equilibrium physics from learned skill;
event-based DR and pushes are disabled, but BAM battery-voltage variation remains.
The calibrated controls offset left hip/ankle by +0.05 rad and right hip/ankle
by -0.05 rad. The standing threshold is based on the learned walking measurement,
not the fixed-control experiment. Shared HOME frames remain unchanged.

CPU regressions cover goal-post collisions, swept goal crossings, invalid strikes,
reward bounds, standing failures, reset isolation, exploration bounds, task
registration, observation layout, and the matching backlash model. Always run
a five-iteration smoke test at 64 environments before a long training run.

## Tracker and deployment contract

Actor input stays `[proprioception(48), twist(3), head_pose(4), body_pose(6)]`.
There are no extra actor ball coordinates or phase flags. Head/body slots retain
small nonzero ranges and exact-zero sampling. The critic alone receives ball
state and reward-history variables.

Hardware needs a ball-and-goal tracker feeding `football_navigation_command`
in `tasks/mdp.py`: ball XY and the desired shot unit vector expressed in the
robot's yaw frame, plus a latched strike flag. The adapter outputs real desired
`[vx, vy, wz]`, with a forward-step cue near the stance and exact zero after a
strike. Simulation uses perfect object tracking and contact sensors.
No camera detector, detector noise/delay model, or hardware strike estimator
is included; those require matching training and transfer validation.

Always use `scripts/export.py` so the observation normalizer is baked into ONNX.
The ONNX alone cannot locate a ball or supply navigation commands. This policy
cannot use the current constant-command episodic `publish` workflow.
