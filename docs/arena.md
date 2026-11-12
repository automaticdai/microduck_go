# Microduck League football arena

Current state: [3v3 tactics and motor audit, 2026-10-09](#3v3-tactics-and-motor-audit--2026-10-09);
earlier resume instructions: [session handoff, 2026-10-04](#session-handoff--2026-10-04).

Two independently controlled ducks share one physical ball and two goals.
The pitch is 1.8 × 1.3 m, with a 22 cm radius center circle. Each kickoff
samples the ball center uniformly over that disk. A goal or 10 seconds ends
a round; falling and leaving the pitch do not. Numerical failures remain
training safety terminations and are reported separately from goals.

The initial policy was trained for individual shots. Arena fine-tuning keeps
its 61D actor observation contract and 14 actions. A ball-and-goal tracker
supplies navigation commands. The actor has no explicit opponent-position
input; the training critic additionally observes opponent position/velocity.
This is a first competition recipe, not yet evidence of learned tactics.

## Three-a-side arena

`Mjlab-FootballArena-3v3-Flat-MicroDuck` places six independently actuated ducks on
a 3.0 × 2.0 m pitch: two attackers and one goal-line defender per team.
Both teams share one ball. Boards keep it in play; goals and 20-second timeouts
reset all six players to mirrored formations. Numerical failures also reset the round.
Attackers use the ball tracker and aim beside the opposing defender; defenders use
the existing keeper tracker and a separately loaded keeper policy. These are fixed
roles using existing policies, not newly learned passing or coordinated tactics.

Replay with team-colored goals, floating triangles, and role labels:

```bash
uv run scripts/play_football_arena_gpu.py --teams \
  --checkpoint path/to/attacker/model_4699.pt \
  --defender-policy path/to/keeper/model_2699.pt \
  --compile-friction --port 8081
```

The same attacker checkpoint controls all four attackers unless `--opponent-policy`
selects a different checkpoint for the two orange attackers. Both defenders use
`--defender-policy`. Open-gap aiming is always enabled in 3v3.

Training updates only the primary blue attacker; its blue teammate and both
defenders stay frozen. Orange attackers use the existing fixed-reference/self-play
selection. Each player retains its own action and velocity history. The actor
remains 61D with 14 actions; the critic additionally sees all five other players.
The learner inherits the base arena's DR, observation noise, BAM actuators and
reward stack; teammates and opponents use clean inference observations.

```bash
# Required smoke test before any longer training run.
uv run scripts/train_arena.py --task Mjlab-FootballArena-3v3-Flat-MicroDuck \
  --from-policy path/to/attacker/model_4699.pt \
  --keeper-policy path/to/keeper/model_2699.pt \
  --num-envs 64 --iterations 5 --run-name 3v3-smoke
```

For sustained fine-tuning, use `--critic-warmup` as with the other arena tasks.
The 3v3 task has its own `football_arena_3v3` experiment directory and requires
`--keeper-policy`. It does not support the two-player `--mode-probs` switch.

## Tactical MAPPO (stage 1)

`scripts/train_mappo.py` trains both blue attackers together on the larger 3v3
pitch. A shared categorical actor selects pursue, support-left, support-right,
cover, or idle at up to 10 Hz. Each selection produces bounded navigation
commands for the frozen 50 Hz motor actors. A round boundary ends a tactical
interval early so advantages never cross a reset. Orange attackers and both
keepers remain frozen. Self-play and learned keeper tactics are later stages.

The tactical actor receives 41 values per player: egocentric ball/player
positions and velocities, goal locations, own state, player role and time
remaining. Its centralized critic receives 87 values including all six robots.
The motor interface remains 61 observations and 14 actions with the checkpoint's
normalizer frozen. Tactical inputs use explicit physical scales. Initial motor
inference is clean for both blue attackers; this tactical baseline is not a
validated perception or hardware deployment stack.

Both attackers receive the same goal/concession reward, discounted ball-progress
potential shaping, and symmetric fall/proximity costs (at most 1 and 0.2 points
per 20-second round, versus 100 per goal). The initial tactical prior chooses
pursuit about 93% of the time to limit early command switching. Terminal potential is
zero, including at finite-horizon timeouts. PPO stores separate log probabilities
and advantages for each attacker. A 24-step rollout means up to 120 motor steps
(2.4 simulated seconds), not the motor trainer's 24-step rollout.

Run on the GCP CUDA server, with the existing attacker and keeper checkpoints:

```bash
uv sync --locked
uv run scripts/train_mappo.py --attacker "$ATTACKER" --keeper "$KEEPER" \
  --num-envs 64 --iterations 5 --strict-numerics --run-name stage1-smoke
uv run scripts/train_mappo.py --attacker "$ATTACKER" --keeper "$KEEPER" \
  --num-envs 64 --evaluate tracker --run-name stage1-baseline
uv run scripts/train_mappo.py --attacker "$ATTACKER" --keeper "$KEEPER" \
  --num-envs 256 --iterations 1000 --critic-warmup 25 --eval-every 100 \
  --run-name stage1
```

Do not launch the sustained run unless the smoke succeeds. Outputs live in
`logs/rsl_rl/football_mappo/`: initialization and motor-checkpoint hashes,
TensorBoard, JSONL metrics, tactical checkpoints, and a separate `tactical.onnx`
(41 inputs → five skill logits). ONNX parity is checked at completion; this
artifact does not replace the 61D motor policy or use its deployment manifest.
Export motor policies only through `scripts/export.py` as usual.

Evaluate a saved tactical checkpoint with `--resume PATH --evaluate policy`.
Evaluation counts the first round of each arena, including standing time only
until that arena finishes. Periodic evaluations pause training and use 64 arenas;
compare scores, concessions, standing and crowding against the tracker baseline.
`--resume` restores model/optimizer/iteration and verifies motor checkpoint hashes;
physics episodes restart. `--iterations` counts additional iterations, including
any initial critic-only warm-up. Inspect rollouts before claiming cooperation.

Numerical failures reset only the affected arena. Its corrupt tactical transition
is excluded from PPO and cuts the GAE trace; the next episode starts from a checked,
finite physics state. `numerics/events.jsonl` identifies the entity, world and
non-finite fields, with pre-reset tensors saved alongside it. Failed recovery still
stops the job. Use `--strict-numerics` to stop on even a recovered event during a
smoke test. Evaluation reports `numerical_rounds` separately.

`latest.pt` is atomically replaced after every completed update, independently of
the numbered checkpoint interval. Resume that file to avoid losing up to 49 updates.
The GPU recovery regression is `scripts/check_mappo_numerics.py` with the same
attacker/keeper arguments and `--num-envs 64 --iterations 5`: it deliberately
injects one NaN into one blue attacker, checks one excluded transition, five finite
updates, the rolling checkpoint and tactical ONNX parity.


## Train on a CUDA server

Run these from the repository root after `uv sync --locked`:

```bash
# Required smoke test before a long run.
uv run scripts/train_arena.py --from-policy path/to/football/model_4999.pt \
  --num-envs 64 --iterations 5 --run-name arena-smoke

# Establish the source policy's deterministic match baseline.
uv run scripts/train_arena.py --from-policy path/to/football/model_4999.pt \
  --num-envs 64 --evaluate-seconds 30 --run-name arena-baseline

# Fine-tune in 2,048 shared arenas for 5,000 iterations.
uv run scripts/train_arena.py --from-policy path/to/football/model_4999.pt \
  --num-envs 2048 --iterations 5000 --run-name arena-selfplay
```

Actor weights and their normalizer transfer together. The critic, optimizer,
and training counters start fresh. Half the worlds retain a fixed reference opponent selected with `--opponent-policy`
(default: `--from-policy`); the other half refresh their frozen opponent at the
`--opponent-update-every` interval (default: 250 iterations).
Both ducks use BAM actuators, and the learner inherits the walking/football
observation noise and domain randomization stack.

Checkpoints are saved every 50 iterations in
`logs/rsl_rl/football_arena/<timestamp>_<run-name>/`. Each `model_N.pt` has a
paired `model_N_opponent.pt`. Resume with the original `--from-policy` and
`--resume path/to/model_N.pt`; `--iterations` means additional iterations.
Keep both checkpoint files together to restore the opponent correctly.

## Goal-first reward revision

The first arena run learned balance but scored infrequently. Revision 2 starts
from its checkpoint 1,000, transferring the actor and normalizer and rebuilding
the critic/optimizer for the new reward targets. Its clock restarts at zero;
4,000 new iterations means roughly 5,000 total iterations of arena experience.

| Signal | Reward |
|---|---|
| Goal / conceded goal | +100 / −100 once |
| Posture and command tracking | At most +0.8 per second combined |
| First supported right-foot strike moving goalward | +3 once |
| Ball speed toward goal after that strike | Up to +3, paid only for new best speed |
| Ball distance toward goal after that strike | +15 per meter of new progress, capped at 1.2 m |
| Approach to the kick stance | +5 per meter of new progress |
| Fallen | −5 per second |
| Timeout without a goal | −5 once |

Ball progress measures distance to the goal mouth, not just forward X motion.
Only upright, supported strikes unlock shot shaping. Actual goals still count
for either team regardless of who touched the ball last. Standing still and
repeatedly tapping the ball cannot repeat the one-time/high-water bonuses.

The 10-second timeout is a finite episode ending. The critic observes remaining
time, while the actor stays 61D. PPO's discount is 0.997 at 50 Hz, extending the
credit horizon to about 6.7 seconds from the previous 2 seconds.

```bash
# Required short check for the revised rewards.
uv run scripts/train_arena.py --from-policy path/to/arena/model_1000.pt \
  --opponent-policy path/to/football/model_4999.pt \
  --num-envs 64 --iterations 5 --run-name arena-v2-smoke

# Four thousand NEW iterations; initialize weights, do not resume old values.
uv run scripts/train_arena.py --from-policy path/to/arena/model_1000.pt \
  --opponent-policy path/to/football/model_4999.pt \
  --num-envs 2048 --iterations 4000 --opponent-update-every 500 \
  --eval-every 250 --run-name arena-v2-from1000
```

`--eval-every 250` evaluates saved actors against the same fixed reference
opponent for 20 simulated seconds in 64 arenas. Reports include goals, conceded
goals, standing fraction, supported strikes, ball speed and timeouts. Training
pauses during these evaluations; inspect `eval_N.log` in the training directory.
Do not directly compare these fixed-opponent evaluations to mixed-opponent
training metrics. `--resume` is for checkpoints from the same reward recipe;
use actor-only `--from-policy` when switching from revision 1 to revision 2.

Monitor `arena_scored`, `arena_conceded`, `arena_standing`, `arena_struck`, the
negative `arena_fallen` and `arena_timeout` costs, and numerical-safety endings.
A fall costs reward but does not restart the round.

## Inspect a trained contender

Export through the normalizer-aware exporter:

```bash
uv run scripts/export.py Mjlab-FootballArena-Flat-MicroDuck \
  --checkpoint-file path/to/arena/model_4999.pt --num-envs 1 \
  --onnx-file logs/arena_policy.onnx

uv run scripts/play_football_arena.py --policy logs/arena_policy.onnx --port 8081
```

Open <http://localhost:8081>. The CPU viewer uses the selected policy for both
teams and shows the cumulative Blue x:y Orange score. Training uses a learner
against a frozen opponent, so the viewer's match results are a separate test.
The CPU arena's transfer from GPU training is still under validation; inspect
both match metrics and visible balance before treating a checkpoint as better.

## Local NVIDIA GPU viewer

For playback with CUDA physics and CUDA policy inference (rather than the CPU
MuJoCo viewer), use the checkpoint directly:

```bash
uv run scripts/play_football_arena_gpu.py \
  --checkpoint logs/arena_v2_model_250.pt --compile-friction --port 8081
```

This uses the training BAM actuator model on the GPU and keeps the pitch,
center-circle spawns, 10-second rounds and cumulative score. The sidebar names
the CUDA device and reports measured playback speed; 1.00× means real time.
Training reward/critic calculations and diagnostic sensors are omitted during
playback. Browser drawing remains WebGL on the client machine.

`--compile-friction` fuses the BAM friction arithmetic without changing its
formula. The first launch compiles kernels before opening the viewer. To measure
simulation throughput without the browser, add `--benchmark-steps 250`.

## Kick-first v3, learned keeper, and the shot eval (2026-09-30)

`scripts/eval_arena_shots.py` is the arena scoreboard. It records the first round of every
arena (no bias toward short rounds), tracks every learner touch to its outcome (goal, wide,
out, blocked, re-touch), and reports fall timing. Use it against three opponents
(`--solo`, `--opponent-role keeper`, default attacker), with `--aim open` and 2048 rounds.
GPU rollouts are not repeatable run to run, so compare rates with their standard errors.

Findings that shaped v3:
- v2 lost the source kick (solo goals 83% → 37%). The cause was fine-tuning against a
  fresh, unfitted critic, not the reward. `train_arena.py --critic-warmup N` freezes the
  actor while the critic fits; always use it when the critic starts from scratch.
- Rounds were decided by about 2 s, then ran dead: the ball left play with no walls and
  near-zero rolling friction. v3 ends the round when the ball goes out and charges a fall
  once (−10) instead of per second.
- The tracker aimed at the goal center, which is exactly where a keeper stands.
  `ArenaCommandCfg.aim='open'` aims at the widest gap beside the opponent's shadow.

Tasks:
- `Mjlab-FootballArena-V3-Flat-MicroDuck` (kick-first-v3): starts from the football kicker
  and stages opponents per arena (empty goal → keeper → attacker, `V3_MODE_STAGES`).
  `train_arena.py --mode-probs S K A` overrides the mix.
- `Mjlab-FootballArena-Keeper-Flat-MicroDuck` (keeper-v1): the learner keeps the −x goal
  against a frozen attacker (`--no-self-play`). Conceding costs −100; a clean round pays +5.

Results (goals within 10 s, 2048 rounds, open aim; source kicker football@4999 shown first):
empty goal 83% → 88% (attacker @3699); vs scripted keeper 30% → 64%; vs attacker
9.8%/7.5% → 12.9%/8.2% (scored/conceded). The learned keeper @1699 lets in 12.5% of v3@2199
shots, against 54.8% for the scripted keeper at the same speed limits.

Viewer: `play_football_arena_gpu.py` takes `--opponent-role {attacker,keeper,solo}`,
`--opponent-policy`, `--aim`, plus boards around the pitch (`--no-boards` to remove them) and a
2 s play-on after goals (`--goal-hold`). Training uses neither by default.

Open issue: StandUp (3000–9000 iterations) rises from sitting but not from lying down.
From face-down it reaches standing height but stays pitched about 34°, so a get-up swap
(`eval_arena_shots.py --getup-policy`) recovers only 3–4% of falls.

### Automatic stand-up recovery and continuation

`--recovery PATH` loads a separate frozen 61D stand-up expert, including its own
observation normalizer, for all six players. After 0.16 s below 6.5 cm trunk height
or tilted more than 60 degrees, that player switches from football to recovery.
All 13 command inputs become zero; proprioception, one-step velocity history,
and the actual previous action remain intact. Football resumes only after 0.4 s
continuously above 9.5 cm, within 30 degrees of upright, with angular speed below
3 rad/s. No action filtering, stand-up teleportation, or per-fall episode reset
is added. A failed attempt keeps trying until the ordinary round ends.

Recovery entries, stable handoffs, and time in recovery are logged separately.
A handoff is a measured posture threshold, not a guarantee of sustained walking.
The recovery expert is frozen: additional MAPPO iterations train tactics under
this controller, not the get-up motion itself. Opponents receive the same
recovery controller, so compare against a baseline evaluated with recovery too.

To migrate the completed v2 tactical checkpoint and add 2,000 iterations:

```bash
uv run scripts/train_mappo.py --attacker "$ATTACKER" --keeper "$KEEPER" \
  --recovery "$GETUP" --resume "$MAPPO_FINAL" --upgrade-recovery \
  --num-envs 64 --iterations 5 --strict-numerics --run-name recovery-smoke
uv run scripts/train_mappo.py --attacker "$ATTACKER" --keeper "$KEEPER" \
  --recovery "$GETUP" --resume "$MAPPO_FINAL" --upgrade-recovery \
  --num-envs 256 --iterations 2000 --eval-every 100 --run-name stage1-v3-recovery
```

`--iterations` counts additional updates after the loaded checkpoint. Starting
from iteration 999 therefore ends at 2999. Migration retains the tactical
weights and optimizer, records the new v3 recovery recipe and expert hash, and
requires the explicit upgrade flag. Subsequent v3 resumes require the same
recovery expert and controller configuration. The exported tactical ONNX still
only outputs the five football choices; playback also needs the frozen motor
experts and recovery controller.

`scripts/check_mappo_getup.py --checkpoint "$MAPPO_FINAL" --recovery "$GETUP"
--output recovery-battery.json` tests 64 controlled face-down, face-up, left-side,
and right-side starts in the arena for eight simulated seconds. It reports
stable handoffs separately by starting orientation and excludes post-reset
states from success counts.

### Keeper-focused motor fine-tuning

`Mjlab-FootballFinishing-Flat-MicroDuck` trains the 61D motor actor on the
3.0 × 2.0 m boarded pitch. It samples both attacker starting sides and mixes
20% empty-goal rounds with 80% rounds against the supplied learned keeper.
Rounds last up to 20 seconds; boards keep rebounds live. It inherits v3's
bounded approach/strike/goal rewards, one-time fall costs, and mature DR.

The measured motivation was 44/64 goals unopposed, 6/64 against the keeper,
and 2/64 in full 3v3. A fixed striker/supporter split reduced crowding but
scored only 1/64. These are diagnostics, not results from the new recipe.

```bash
uv run scripts/train_arena.py --task Mjlab-FootballFinishing-Flat-MicroDuck \
  --from-policy "$ATTACKER" --keeper-policy "$KEEPER" --recovery-policy "$GETUP" \
  --num-envs 64 --iterations 5 --critic-warmup 2 --run-name finishing-smoke
```

Initialization loads only the actor and its normalizer, builds a fresh critic,
and resets the iteration/curriculum counters. The keeper stays frozen; self-play
is disabled. `--recovery-policy` swaps only the frozen opponent to recovery:
the learning attacker always executes its sampled action, preserving PPO's
action/log-probability contract. Evaluate candidate motors in the original 3v3
arena with deployment recovery enabled for everyone before adopting them.
Use `scripts/export.py` for motor ONNX export as usual.

### Recovery playback and handoff validation

The GPU viewer requires an explicit recovery checkpoint; football weights alone
cannot stand fallen players back up. Use the same expert as the GCP run:

```bash
uv run scripts/play_football_arena_gpu.py --teams \
  --checkpoint "$ATTACKER" --defender-policy "$KEEPER" \
  --recovery-policy "$GETUP"
```

The overlay labels each player `Recovery`, `Attacker`, or `Defender`, and the
panel reports whether recovery is enabled. The expert loads its own normalizer;
only its command slots are zeroed. Both two-player and 3v3 playback support it.
This viewer uses navigation commands, not the learned MAPPO tactical checkpoint.

`check_mappo_getup.py` counts a 0.4-second stable detection, which does **not**
prove a successful return to play. For that, run the bounded paired battery:

```bash
uv run scripts/check_recovery_handoff.py --checkpoint "$MAPPO_FINAL" \
  --scenario isolated --output artifacts/recovery/isolated
# Repeat with --scenario board and --scenario contact.
# --handoff-mode zero or ramp isolates football idle and command onset.
```

Each condition contains 64 orientation-balanced starts, comparing normal
football handoff against retaining the recovery expert. Initial poses match;
inherited domain randomization is independently sampled. The proposed gate is
at least 90% standing within five seconds and remaining upright continuously
for the next three seconds. Movement after handoff and numerical failures are
reported separately. Board and contact starts are controlled stress cases,
not samples of naturally occurring match falls. Keeping the recovery expert
forever is a diagnostic control, not a playing solution.

The first recovery battery found 64/64 isolated get-ups but only 36/64 durable
football handoffs, versus 64/64 when the expert retained control. Near boards:
2/64 versus 63/64. Contact starts: 26/64 versus 64/64. A one-second command ramp
failed (22/64), so no ramp is enabled in playback.

`Mjlab-FootballReturnToPlay-Flat-MicroDuck` practices the measured handoff
boundary. Half its resets sample `handoff_states.pt` emitted by the battery;
half retain normal spawns. The bank carries root pose/velocity, servo state,
and previous expert action. This samples the observed physical state; it does
not restore the source simulator's actuator or observation-delay internals.
The 61D actor, unfiltered action interface, BAM and mature randomization remain.
Rounds are eight seconds, 80% empty goal and 20% keeper; 0.2 seconds continuously
outside the upright region ends the attempt with the existing v3 fall cost.

```bash
uv run scripts/train_arena.py --task Mjlab-FootballReturnToPlay-Flat-MicroDuck \
  --from-policy "$ATTACKER" --keeper-policy "$KEEPER" --recovery-policy "$GETUP" \
  --recovery-starts artifacts/recovery/isolated/handoff_states.pt \
  --num-envs 64 --critic-warmup 2 --iterations 5 --run-name return-smoke
```

Only extend after the smoke/export checks. Evaluate a candidate with
`check_recovery_handoff.py --attacker candidate.pt` and the original tactical
checkpoint as `--checkpoint`; the opposing attacker remains fixed. Passing
training metrics alone does not clear the recovery or match-scoring gates.

CPU playback accepts the same option with an **ONNX** file:
`uv run scripts/play_football_arena.py --policy football.onnx --recovery-policy getup.onnx`.
Export both through `scripts/export.py` so each includes its own normalizer.
The CPU status panel shows `Football`/`Recovery` separately for Blue and Orange;
its adapter calls the same `ArenaRecovery` class as GPU playback and training.

CPU parity caveat from this audit: the existing arena disabled the M6 current
limit, and upstream `bam.mujoco.MujocoController` subtracted DOF friction using
joint IDs. The arena now retains the model's current limiter and uses
`CpuBamController` to subtract friction by DOF ID, matching mjlab. A floating
base regression test verifies the mapping against MuJoCo's actual constraints.
These corrections do not yet establish reliable CPU get-up behavior: a small
expert-only four-orientation test still showed short upright intervals. Use the
GPU battery for the reported recovery gate; treat CPU physical recovery as
unvalidated even though ONNX numerical parity passes.

Use `--scenario collision` for an incoming opponent with a contact sensor. The
older `contact` case is only a close-neighbor start: its recorded sample did
not show actual robot-to-robot contacts. The collision report includes
`measured_collision_trials`; compare recovery conditional on contact, since a
stationary control and a moving player encounter different numbers of impacts.

The bounded 200-update return-to-play pilot (512 environments, 25 critic warm-up
updates) did **not** clear the gate and was not promoted. Durable isolated
handoffs fell from 36/64 to 25/64; board starts improved from 2/64 to 27/64.
Incoming-opponent starts scored 12/64 versus 14/64, with actual impacts in
23 versus 30 trials, respectively. Empty-goal scoring fell from 44/64 to 26/64;
keeper-only scoring fell from 6/64 to 3/64. Its final match battery had two
numerical resets. Retain the original motor; this recipe is an experiment,
not a validated recovery solution. Full evidence and reproducible local
playback launchers are in `artifacts/recovery-handoff/` (ignored run artifacts).

### Isolated recovery-to-walking training (2026-10-04)

`Mjlab-RecoveryWalk-Flat-MicroDuck` removes football entities and rewards and
builds on the velocity recipe with the source actor's groundcontact BAM model.
75% of resets use recorded recovery endings; the remainder use normal locomotion
resets. Eight command buckets practice idle, forward, backward, lateral, and
turning, with an explicit stop at 3 s and restart at 4 s. Episodes last 6 s;
spending 0.2 s below the handoff height/tilt gate ends a failed attempt.

`scripts/train_recovery_walk.py` loads only the actor and its normalizer, uses a
fresh critic, and resets iteration/curriculum counters. Mature CoM randomization
is retained. Actor updates use fixed learning rate 1e-4 and low action-rate cost.
The source recovery expert remains unchanged.

Before a long run, run 64 environments for 5 iterations and export through
`scripts/export.py`. Long runs require a separate `--eval-bank`; exact duplicate
states are rejected. Evaluate every 50 actor updates, crossing every held-out
pose with seven fixed commands using a separate simulation seed. Reports count
continuous upright time, signed movement, stopping speed, restart movement, and
numerical failures. A reset fails the original trial. Passing requires at least
90% upright-and-command success in **each** command bucket over the first 3 s.
Stopping/restarting over 6 s are additional diagnostics, not covered by that gate.
These bank resets restore physical state and previous action, but do not replay
actuator/observation-delay caches: full live recovery handoffs still require a
separate evaluation before promotion.

```bash
uv run scripts/train_recovery_walk.py --from-policy ORIGINAL_ATTACKER.pt \
  --bank TRAIN_STARTS.pt --eval-bank HELDOUT_STARTS.pt \
  --num-envs 64 --iterations 5 --run-name smoke
# After smoke and normalized ONNX export pass:
uv run scripts/train_recovery_walk.py --from-policy ORIGINAL_ATTACKER.pt \
  --bank TRAIN_STARTS.pt --eval-bank HELDOUT_STARTS.pt \
  --num-envs 2048 --iterations 1000 --critic-warmup 25 --eval-every 50 \
  --run-name isolated-handoff
# Local GPU playback from measured recovery endings:
uv run scripts/train_recovery_walk.py --from-policy CHECKPOINT.pt \
  --bank TRAIN_STARTS.pt --num-envs 16 --play
```

GPU playback is the reference visual check. Training/evaluation do not promote
checkpoints into the football arena or change the existing MAPPO run.

#### Local GPU audit of the isolated run

The 2026-10-04 RTX 5080 audit uses `scripts/check_recovery_walk.py` to replay
all 64 held-out poses under seven commands and record pose index 0 for forward
and leftward commands. Both checkpoints use seed 2026. Videos and per-command
reward mass are saved under `artifacts/recovery-walk/local-gpu/update{300,1000}`.
These are measured-pose resets, not full live get-up transitions.

Update 300 averaged 0.532 m forward and 0.381 m leftward travel in the first
3 s. Update 1000 averaged 0.196 m forward and effectively zero leftward travel.
Nevertheless, mean weighted reward per alive step increased from 6.38 to 7.81
for forward and from 6.04 to 8.55 for leftward. Both local evaluations had zero
numerical failures. The early actor also failed to stop cleanly: average linear
speed during the final half-second of the stop window was about 0.20 m/s for
both moving commands.

The velocity template gives a stationary robot exp(-0.1²/0.1) = 90.5% of the
linear tracking reward under a 0.1 m/s command. Its linear reward includes
vertical velocity error, and angular tracking includes roll/pitch rates, so
stepping oscillations reduce tracking rewards even when planar movement improves.
Later checkpoints collect more posture, upright, head, and angular-tracking
reward while moving less. This supports correcting the reward balance and
separating commanded-axis tracking from gait oscillations before another run;
it does not establish that an untested replacement reward will work.

On this WSL host, Warp loaded the wrong Linux CUDA driver by default although
PyTorch could use the RTX 5080. A process-local library path resolved it:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib MUJOCO_GL=egl \
  uv run scripts/check_recovery_walk.py \
  --checkpoint artifacts/recovery-walk/training/model_324.pt \
  --bank artifacts/recovery-walk/heldout_starts.pt \
  --output artifacts/recovery-walk/local-gpu/update300 --label '300 updates'
```

Use `model_1024.pt`, `update1000`, and label `1000 updates` for the final actor.
No reward changes or additional training were launched during this audit.

#### Recovery-walk v2 pilot

The v2 recipe starts from update 300 of the first isolated run, loads actor and
normalizer only, fits a fresh critic for 25 updates, then performs a bounded
200-update pilot at 2048 environments. Held-out checks remain every 50 updates.
No longer run is warranted unless movement improves while stability exceeds 90%.

`RecoveryWalkTracking` measures planar velocity and yaw separately, excluding
vertical bobbing and roll/pitch from command error. A 0.15 s average smooths only
reward measurements; actions and the 61D observation contract are unchanged.
Tracking widths are 0.10 m/s and 0.25 rad/s. Bounded signed progress in the
commanded direction pays zero at rest, negative for wrong-way movement, and
saturates at commanded speed. It pays zero under a zero command. Tracking terms
still discourage overspeed and off-axis drift. History resets per environment.
Posture, head alignment, and gait regularization retain smaller supporting weights.

Before cloud training, matched local GPU replays ranked update 300 above the
nearly stationary update 1000 under v2: forward reward 6.60 versus 5.12 and
leftward 6.30 versus 4.66. This validates the intended ranking for those two
recorded policies, not successful learning. A preliminary version using tighter
tracking alone failed that ranking check and was not trained.

Cloud checkout: `/mnt/data/microduck-recovery-walk-v2-20261004`.
Local provenance and preflight reports: `artifacts/recovery-walk-v2/`.
The cloud launcher checks the five-update smoke result and normalized ONNX
parity before starting the pilot. The first run and original MAPPO job are
kept separate.

#### Focused-practice v3 pilot

The v2 local GPU audit confirmed the actor receives signed backward/right/yaw
commands. Backward and rightward travel remained around 3 mm in 3 s; turn angles
were roughly 0.13 and 0.08 rad. Feet remained largely planted. Forward/left
movement survived, but stop-window speeds were still about 0.21–0.25 m/s.

V3 keeps v2's rewards and changes practice allocation: idle 10%, forward 15%,
backward 20%, left 10%, right 15%, left turn 15%, right turn 15%. Training uses
8 s episodes with motion 0–3 s, stop 3–5 s, and restart 5–8 s. Evaluation retains
the original 0–3/3–4/4–6 s schedule so its results remain comparable. An optional
`--initial-action-std-floor .08` raises exploration once at actor initialization;
the loaded leg standard deviations were roughly 0.04–0.06. Deterministic policy
means and normalization are preserved. This floor is training-only, not an action
filter, and normal PPO updates can subsequently change the standard deviations.

The new full evaluation gate additionally requires mean stop speed <=0.03 m/s
and <=0.15 rad/s during 3.5–4 s, continuous uprightness for 6 s, and restart
travel at least two-thirds of the initial 3 s threshold. The old first-3-second
criterion remains available as `movement_gate_passed`. A reset always fails the
original trial. Unit tests cover perfect sequences, ignoring stop, and resets.

Start from the v2 `model_224.pt`, fit a fresh critic for 25 updates, then run a
bounded 200-update pilot at 2048 environments, evaluating every 50. The source
workspace remains `/mnt/data/microduck-recovery-walk-v2-20261004`; v3 runs in
`/mnt/data/microduck-recovery-walk-v3-20261004`, with local provenance under
`artifacts/recovery-walk-v3/`.

The user authorized pausing the older MAPPO run. PID 770053 was verified against
its command line and paused with SIGSTOP; `kill -CONT 770053` resumes it while
that same process still exists. It retains GPU memory but no longer executes
training, and the GCP VM remains running.

## Session handoff — 2026-10-04

**Work is stopped at the user's request. No new experiment is queued.** The v3
pilot completed normally; it did not solve backward/rightward movement, turning,
or stopping. No candidate has been promoted into football play or deployed to
hardware. Keep the get-up specialist unchanged.

### Last verified operational state

- GCP host: `136.85.147.20`. The VM is **still powered on and billable**; it was
  not shut down as part of dismissing this session.
- No active training. GPU utilization was 0% at the handoff check.
- Older MAPPO process **770053 is paused** (`Tl`, SIGSTOP), retaining about
  4.3 GB of GPU memory. Resume only when wanted, after verifying the PID still
  belongs to `scripts/train_mappo.py --run-name stage1-v3-recovery-to5000`.
  The resume command is `kill -CONT 770053`; this is not a request to resume it.
- v3 exited with code 0 after **25 critic warm-up + 200 actor updates**. All
  225 logged iterations had zero numerical terminations. Smoke and normalized
  ONNX parity passed (maximum absolute error 4.77e-7); 24 related tests passed.
- Source changes remain **uncommitted**, including work from earlier in the
  conversation. Preserve the entire working tree; no commit/push was requested.

### Final v3 held-out results

64 recorded recovery endings per command, pose-bank seed 7, simulation seed
2026. Initial movement is measured over 0–3 s; stop over 3.5–4 s; restart over
4–6 s. Counts below are successful trials out of 64.

| Command | Upright for 6 s | Initial movement/idle | Stop | Restart/idle | Complete sequence |
|---|---:|---:|---:|---:|---:|
| Idle | 64 | 64 | 64 | 64 | 64 |
| Forward | 63 | 63 | 0 | 63 | 0 |
| Backward | 63 | 0 | 63 | 0 | 0 |
| Left | 62 | 64 | 0 | 62 | 0 |
| Right | 63 | 0 | 63 | 0 | 0 |
| Turn left | 60 | 0 | 60 | 0 | 0 |
| Turn right | 64 | 0 | 64 | 0 | 0 |

Forward/left stop-window speeds remain about **0.20/0.24 m/s**, despite the
zero command. Backward/rightward travel is about **1.27/0.32 cm in 3 s**.
Yaw travel is about **0.105/0.080 rad**, below the 0.6 rad movement threshold.
No checkpoint at updates 50, 100, 150, or 200 passed the full acceptance gate.
The extra weak-command practice, longer training stop window, and exploration
floor did not produce the missing behaviors in this pilot. Another identical
pilot is not supported by these results.

These are **motor-control tests, not goal-rate measurements**: no ball or
opponents are present. Bank resets restore pose, velocity, joints, and previous
action, but not the full actuator/observation-delay history of a live get-up.
Full live recovery handoffs still need separate validation after basic command
following passes.

### Artifacts and provenance

Local experiment directories are ignored by Git; the source and this document
are in the working tree. The completed v3 evaluations and final checkpoint were
copied locally during this handoff.

| Experiment | Local artifacts | Cloud checkout / final run |
|---|---|---|
| v1, 1000 updates | `artifacts/recovery-walk/` | `/mnt/data/microduck-recovery-walk-20261004`; run `2026-10-04_11-10-16_isolated-handoff-1000` |
| v2, 200 updates | `artifacts/recovery-walk-v2/` | `/mnt/data/microduck-recovery-walk-v2-20261004`; run `2026-10-04_15-24-50_v2-pilot200` |
| v3, 200 updates | `artifacts/recovery-walk-v3/` | `/mnt/data/microduck-recovery-walk-v3-20261004`; run `2026-10-04_17-11-05_v3-pilot200` |

Each cloud run is under `logs/rsl_rl/recovery_walk/` in its checkout. V2/v3
final checkpoints are `model_224.pt` (25 warm-up + 200 actor updates, zero-based
iteration). V1 update 300 is `model_324.pt`; v1 final is `model_1024.pt`.

- Latest local checkpoint: `artifacts/recovery-walk-v3/training/model_224.pt`.
- Latest reports: `artifacts/recovery-walk-v3/training/eval_{0,50,100,150,200}.json`.
- Run records and source hashes: `artifacts/recovery-walk*/run.json`.
- Original motor: `artifacts/recovery-handoff/attacker-4699.pt`; frozen recovery:
  `artifacts/recovery-handoff/getup-12997.pt`.
- Recorded banks: `artifacts/recovery-walk/{train_starts,heldout_starts}.pt`.
- V1 local GPU recordings: `artifacts/recovery-walk/local-gpu/update{300,1000}/`.
- V2 final local recording: `artifacts/recovery-walk-v2/local-gpu/`.
- V2 weak-command recordings and actor-input audit:
  `artifacts/recovery-walk-v3/diagnose-{backward-right,turns}/`.
  These directories contain **v2 baseline** rollouts, not the v3 final policy.
- **V3 final local GPU playback has not yet been run.** Its final results above
  come from GCP's held-out evaluation.

Primary code: `microduck_recovery_walk_env_cfg.py`, the `RecoveryWalk*` terms in
`mdp.py`, `scripts/train_recovery_walk.py`, and `scripts/check_recovery_walk.py`.
Regression coverage: `tests/test_recovery_walk_cfg.py` and
`tests/test_recovery_walk_evaluation.py` plus existing related cfg tests.
The earlier v2 background collection process did not survive; collection and
local playback were subsequently completed explicitly. No follow-up process is
being relied on for future work.

### Recommended next experiment — not started

1. Obtain and verify the original walking expert's exact checkpoint/provenance.
   Compare its 61D observation normalization and command response with the
   football-derived motor. Do not seed walking from the command-narrow stand
   expert; the normalizers are incompatible for that purpose.
2. Evaluate that walking expert on the same groundcontact/BAM robot, recorded
   banks, seven commands, and stop/restart battery. Inspect **local GPU playback**.
   Compare with v2/v3; also inspect the v3 final locally if needed.
3. If the walking expert passes or clearly improves the missing behaviors, use
   it as the actor starting point for a new recovery-handoff recipe. If both
   policies fail, investigate interface/normalization/physics before more PPO.
4. Any new recipe still needs the 64-environment, five-update smoke test and
   normalized ONNX export before a bounded training run. Keep football/opponents
   out until movement, stability, stopping, and restarting pass.

Local GPU access on this WSL machine needs execution outside the sandbox plus
`LD_LIBRARY_PATH=/usr/lib/wsl/lib`. Offscreen recording uses `MUJOCO_GL=egl`.
The GPU is an RTX 5080. PyTorch alone could see it while Warp loaded the wrong
Linux driver until the library path was set; no system driver was modified.

Example replay of the archived v3 final (this command has **not** been run):

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib MUJOCO_GL=egl \
  uv run scripts/check_recovery_walk.py \
  --checkpoint artifacts/recovery-walk-v3/training/model_224.pt \
  --bank artifacts/recovery-walk/heldout_starts.pt \
  --output artifacts/recovery-walk-v3/local-gpu --label 'v3 final' \
  --video-commands backward right
```

GCP reconnection, if needed:

```bash
ssh -T -i /home/yfrl/.ssh/google_compute_engine \
  -o HostKeyAlias=compute.462629909934299217 -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/home/yfrl/.ssh/google_compute_known_hosts \
  -o BatchMode=yes yfrl@136.85.147.20
```

## 3v3 tactics and motor audit — 2026-10-09

Evaluation only; no training was launched and no checkpoint was promoted.
`scripts/check_hybrid_motor.py` plays 256 first rounds of 20 s in the 3v3 arena.
Orange keeps attacker 4699, keeper 2699 and the get-up expert 12997; blue
changes only its tactics (`--tactics`) or attacker motor (`--blue-motor`).
Motors are byte-identical (SHA-256) to the GCP originals. Goals happen in only
a few percent of rounds, so **64-round evaluations cannot rank tactics**: use
≥256 rounds and at least two seeds.

### Tactics (football motor, seeds 42 + 7, 512 rounds pooled)

| Blue tactics | For | Against |
|---|---:|---:|
| MAPPO stage1-v3 `model_2999` | 5 | 35 |
| Pursuit (`tracker`) | 14 | 13 |
| Nearest pursues, other covers | 15 | 20 |
| Nearest pursues, other supports | 11 | 20 |

The MAPPO checkpoint is a regression against plain pursuit. It idles about
48% of the time and pursues only 10–18% (`artifacts/goal-diagnosis/`). Its
periodic 64-round evaluations (0–4 goals each) could not reveal this. Scripted
role splits do not beat pursuit. Blue attackers still fall about three times per
round, even when only one blue player chases the ball. **The falls come from the
football motor itself, not teammate collisions.**

### Walking expert

`walking_model_3000.pt`, the source of the football lineage (GCP
`~/microduck_rl/checkpoints/`, SHA-256 `4b5c9b72…`), has a compatible walking
normalizer. On the recovery-walk battery from held-out recovery endings, it never
falls and stops cleanly. Below 0.2 m/s it barely moves, because the velocity
template pays a standing robot about 90% of tracking reward at 0.1 m/s.
`--command-scale` (new, default 1) multiplies the battery commands; thresholds
are unchanged.

| Recovery-walk battery (64 trials/command) | Upright 6 s, worst command | Complete sequence, worst command |
|---|---:|---:|
| Walker, 1× commands | 64 | 0 (backward/left/right) |
| Walker, 3× commands | 64 | 62 |
| Football attacker 4699, 3× commands | 0 (backward) | 0 |
| Football attacker 4699, idle | 26–27 | 15–16 |

In 3v3, the walker keeps blue attackers upright 97–99% of the time, versus about
52%. However, it cannot strike: pursuit with the walker scores 0/256 and
concedes 22. A hybrid uses the walker away from the ball, with the football
motor within 0.35/0.50 m and a 1 s settle after get-up. It scored 7/256 and
conceded 17, so it is not better than pure football. Results:
`artifacts/walking-expert/` and `artifacts/hybrid-motor/`.

### Implications for the next step

- Use pursuit, not the stage1-v3 MAPPO checkpoint, as the playing and comparison
  baseline. Any tactical retraining needs evaluation of at least 256 rounds;
  the current reward and evaluation scale did not learn to pursue.
- The ceiling is the football motor: about 50% of each round spent in recovery.
  The walker shows that the same robot and BAM model can stay upright under
  the same contact conditions. A promising candidate is a striker trained from
  the walker. It would preserve the walker's stability and command response,
  then add striking only within its operating command range (≥0.2 m/s or a
  near-ball kick command), with mature DR and the fall-cost recipe. That
  is a proposal, not a tested result.

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run scripts/check_hybrid_motor.py \
  --checkpoint mappo-2999.pt --walker walking_model_3000.pt \
  --attacker attacker-4699.pt --keeper keeper-2699.pt --recovery getup-12997.pt \
  --blue-motor football --tactics tracker --seed 42 --output tracker-football.json
```
