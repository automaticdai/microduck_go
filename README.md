# Microduck Go

**An arena for trained Microducks to face off.**

[Microduck Go](https://github.com/automaticdai/microduck_go) brings learned robot
skills into a shared arena: train a duck, enter it in a match, and watch it
compete against another duck. Football is the first matchup — **two ducks, two
goals, one ball** — with robot fights and other competitive challenges as the
broader direction.

The contenders are [Microducks](https://github.com/pollen-robotics/microduck),
~800 g, ~25 cm tall bipedal robots with 14 servos. The project builds on the
Microduck RL training stack: [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp), PPO, and [BAM](https://github.com/Rhoban/bam) actuator physics.
Training, evaluation, and ONNX export support the arena and provide a path to
running learned skills on real robots.

## Current state

| Capability | Status |
|---|---|
| Single-duck football | Implemented: approach a tracked ball, shoot into a goal, and stay standing. A trained checkpoint has been visually reviewed locally. |
| Two-duck football arena | In development: a shared pitch, opposing goals, one physical ball, and independent policy control. Match behavior is still being validated. |
| Competitive tactics and self-play | Planned: the current football policy learns individual shots, not defense, tackling, or opponent-aware strategy. |
| Robot fights and additional game modes | Project direction; not implemented yet. |

Trained weights are not bundled with the repository. The instructions below
use checkpoint paths you provide. See the [football guide](docs/football.md)
for the existing training task and its observation and scoring contracts.

## Get started

Install [uv](https://docs.astral.sh/uv/) and use Python 3.12. Training and the
mjlab policy viewer require a CUDA GPU; exported policies can run in CPU MuJoCo.

```bash
git clone https://github.com/automaticdai/microduck_go.git
cd microduck_go
uv sync --locked

# Discover the available training environments.
uv run list-envs

# Watch an existing football contender in the browser viewer.
uv run play Mjlab-Football-Flat-MicroDuck \
  --checkpoint-file path/to/football/model_4999.pt --num-envs 1 --viewer viser
```

Open the local URL printed by the viewer, normally `http://localhost:8080`.
This command runs the established single-duck approach-and-shoot task. The
competitive arena is the next step and is not yet a validated quickstart.

On ARM machines (DGX Spark / GB10, Jetson), set `UV_HTTP_TIMEOUT=600` before
the first sync to allow time for the CUDA wheels to download.

## From skill to competition

1. **Train a contender.** Start with walking and balance, then train an arena
   skill such as approaching a ball and shooting.
2. **Evaluate the policy.** Check standing, task success, and visual behavior;
   a higher training reward alone does not establish match readiness.
3. **Enter the arena.** The arena under development puts independently controlled
   ducks in the same physics simulation, with shared objects and match rules.
4. **Improve through competition.** Opponent-aware training and self-play are
   future work built on those match outcomes.

The Python package remains `mjlab_microduck`, and existing `Mjlab-…-MicroDuck`
task IDs and command names remain valid after the project rename.

For remote training, add `--hf-jobs` to the standard `train` command; see the
[Hugging Face Jobs guide](scripts/hf/README.md). Football's training helper can
also run directly on a remote CUDA server.

## Train a football contender

`Mjlab-Football-Flat-MicroDuck` trains the duck to approach a tracked ball,
line up a right-foot shot into a physical goal, and remain standing afterward.
The goal is 40 cm wide and 30 cm high, positioned 75 cm beyond the ball.
A score requires the whole ball to cross between the posts and below the bar.
The matching backlash task is `Mjlab-Football-Flat-Backlash-MicroDuck`.

Start from a compatible 61D walking checkpoint. The helper transfers the actor
and its observation normalizer, with a fresh critic, optimizer, and curriculum.
Checkpoints are not bundled with this repository. Run the smoke test before
starting a long run, on either a local CUDA GPU or a remote GPU server:

```bash
uv sync --locked

# Validate this walking initialization: 64 environments, five iterations.
uv run scripts/train_football.py --from-walking path/to/walking/model_3000.pt \
  --num-envs 64 --iterations 5 --eval-every 5 --run-name football-smoke

# Train for 5,000 iterations, checking standing and approach every 100.
uv run scripts/train_football.py --from-walking path/to/walking/model_3000.pt \
  --num-envs 4096 --iterations 5000 --eval-every 100 --run-name football
```

Runs and checkpoints are saved under `logs/rsl_rl/football/<timestamp>_<run-name>/`.
Evaluation checks stop training on a substantial standing or approach regression.
To keep training while recording those failures, add `--continue-on-regression`.
This changes the evaluation stop policy; physical balance checks remain active.

```bash
# Resume full training state. --iterations counts ADDITIONAL iterations:
# checkpoint 999 + 4,000 more finishes at checkpoint 4999 (5,000 total).
uv run scripts/train_football.py --resume path/to/football/model_999.pt \
  --num-envs 4096 --iterations 4000 --eval-every 100 --run-name football-resume

# Watch repeated approach-and-shoot episodes in an interactive browser viewer.
# Open the localhost URL printed by the viewer (normally port 8080).
uv run play Mjlab-Football-Flat-MicroDuck \
  --checkpoint-file path/to/football/model_4999.pt --num-envs 1 --viewer viser

# Evaluate 192 approach episodes; add --idle for standing or
# --near-probability 1 for shots starting beside the ball.
uv run scripts/eval_football.py --checkpoint-file path/to/football/model_4999.pt \
  --output logs/football_eval.json --video logs/football_eval.mp4

# Export through the normalizer-aware export path.
uv run scripts/export.py Mjlab-Football-Flat-MicroDuck \
  --checkpoint-file path/to/football/model_4999.pt --num-envs 1 \
  --onnx-file football.onnx
```

The actor retains the shared 61D observation contract. Simulation supplies a
ball-and-goal tracker that converts object positions into navigation commands;
hardware needs an equivalent tracker and strike detector. The ONNX alone does
not provide vision or navigation, and this task cannot use the constant-command
`publish` workflow. See [the football guide](docs/football.md) for scoring,
standing checks, physics validation, and evaluation details.

## Foundation skills

`uv run list-envs` prints the live registry. Flat/Rough variants exist where noted.

<!-- SHOWCASE GRID — one short GIF per task family (sim or real), 3 per row.
     Priority order if you only record a few: Velocity, VelStand (fall+recover),
     Roulade, SitStand, Rollers/Swizzle, BallKick. -->

| Task id | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | Walking with velocity commands + head-pose commands |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking + fall recovery in one policy |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Stand up from face-down/face-up/sitting, then hold the stand + body-pose control |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | flat/rough | Commanded sit ↔ stand in one policy, gently, head commandable |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the mouth tip, return to stand |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm / 15 g ball forward (actor is ball-blind) |
| `Mjlab-Football-Flat-MicroDuck` | flat | Approach a tracked 70 mm ball and shoot into a physical goal; [training and tracker contract](docs/football.md) |
| `Mjlab-Roulade-Flat-MicroDuck` | flat | Forward roll over the head, land back on the feet |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller-skate velocity tracking (passive wheels under the feet) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | Classic symmetric swizzle skating |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while gliding on rollers |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Glide down slopes on rollers |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | flat | Stand up from the ground onto the wheels |
| `Mjlab-Spin-Flat-MicroDuck` | flat | Fast spin in place on rollers |

At deployment the runtime hot-swaps these policies (walk / recover / trick)
behind a shared 61-dimensional observation contract, so any of them can take
over the robot at any moment. `scripts/infer_policy.py` rehearses exactly that:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

Keyboard-driven (velocity commands, `G` ground pick, `Y` sit/stand, `R` roulade,
`K`/`L` kicks); `--debug`, `--save-csv`, `--record` support sim2real comparisons.
The servos are simulated with the same BAM M6 XL330 model the policies are
trained against (voltage control + load-dependent friction, via
`bam.mujoco.MujocoController`); `--vin` / `--vin-drop-gain` / `--kp-fw` pin the
training DR ranges to one value, `--no-bam` falls back to the XML PD actuators.

### Backlash variants

Every main task has a **Backlash** twin that trains on a model with ±1° of gear
play (2° total) in series with each of the 14 servo joints: insert `-Backlash`
before `MicroDuck` in the task id, e.g. `Mjlab-Velocity-Flat-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos`/`joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/mjlab_microduck/tasks/backlash.py`.

## Actuator model

All tasks use the [BAM](https://github.com/Rhoban/bam) M6 actuator model for
the Dynamixel XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-env domain randomization on battery voltage, voltage sag
under load, command delay, and friction magnitude
(`FrictionDRBamActuator` in `src/mjlab_microduck/actuator/`).

At this scale — tiny servos driving a ~800 g biped — actuator fidelity is most
of the sim2real gap, which is why the actuator is modeled down to its voltage
control law instead of an ideal PD.

## Robot models

MJCF models live in `src/mjlab_microduck/robot/microduck/` and are exported
from Onshape with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot),
one `config_mjcf_*.json` per model:

| XML | Used by |
|---|---|
| `robot_walk.xml` | Velocity (stripped trunk/head contacts — falling is cheap) |
| `robot_groundcontact.xml` | VelStand, StandUp, SitStand, GroundPick, BallKick, Football, Roulade (curated collision set for the parts that touch the floor — body can physically lie on the ground; formerly `robot_allcollisions.xml`) |
| `robot_groundcontact_rollers.xml` | Roller tasks (passive wheels) |
| `robot_allcollisions.xml` | True full-collision model — every part has a collision geom. Tasks needing full-body collision coverage |
| `robot_*_backlash.xml` | Backlash task variants (generated by `add_backlash.py`) |

`scene*.xml` files wrap the robots with a floor + keyframes (STAND/SIT/FOLD)
for quick viewing and for `infer_policy.py`.

<!-- IMAGE — side-by-side render: walk model vs rollers model (or a collision-geom
     visualization). One image here makes the model-variant story instant. -->

## Project structure

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── train_cli.py                      # `train` script (identical to mjlab's)
├── train_hook.py                     # intercepts `train ... --hf-jobs`
└── hf_jobs.py                        # Hugging Face Jobs submission
```

Conventions worth knowing:

- The observation layout is shared across every policy (61-dim actor obs:
  48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`), which
  is what makes runtime policy hot-swapping possible. Envs that don't use a
  command slot zero-pad it rather than dropping it.
- Unactuated joints are all named `passive_*` (roller wheels, backlash
  hinges); actuators, joint observations and pose rewards select servo joints
  with `^(?!passive_).*`.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each
  env cfg file.
- Joint layout (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll),
  9–13 right leg.
- The exporter bakes the observation normalizer into the ONNX graph — always
  deploy ONNX produced by `scripts/export.py`, never a hand-converted
  checkpoint, or the policy sees unnormalized observations at runtime.

[AGENTS.md](AGENTS.md) documents the env-building workflow and the reward-design
rules learned across the project (also aimed at AI coding agents working in
this repo).

## Deploying skills to a real duck

`uv run publish` puts a policy on the Hugging Face Hub in the shape the robot's
daemon loads: one `policy.onnx` with the observation normalizer baked in, a
`manifest.json` following schema 2 of the
[microduck policy manifest](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md),
and a README saying how to run it. Anyone with a microduck can then install it
with one command, no daemon release needed.

```bash
# From a wandb run — exports through the one safe path, then uploads
uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000 \
    --repo <user>/microduck-polite-bow --kind episodic --duration-s 4.0 \
    --description "Bows from a two-foot stand and comes back up."

# From an ONNX you already exported (validated, not re-exported)
uv run publish --onnx output.onnx --repo <user>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5 --twist-help "[flag, side, 0]"

# A new gait for a slot
uv run publish --onnx output.onnx --repo <user>/microduck-my-walk --kind perpetual --slot walk

# See what would be uploaded without touching the Hub
uv run publish --onnx output.onnx --repo <user>/microduck-bow --kind episodic --duration-s 4.0 --dry-run
```

Then on a robot:

```bash
sudo robotctl policy add polite-bow <user>/microduck-polite-bow   # episodic: length comes from the manifest
sudo robotctl policy add flamingo <user>/microduck-flamingo --hold 5   # held pose: you pick how long
sudo robotctl policy load walk <user>/microduck-my-walk                # gait: into the walk slot
robotctl robot do polite-bow
```

What `--kind` means, and what each needs:

- **episodic** — runs for `--duration-s` and returns itself to a standing pose
  (kicks, roulade, a bow). Add `--chain` if holding the button should repeat it.
- **perpetual** — runs until told otherwise. Two shapes:
  - a **gait** (a new walk or stand): add `--slot walk` (or `stand`) and
    nothing else; the owner installs it with `robotctl policy load walk <repo>`.
  - a **held pose** (the flamingo): give `--unwind-s`, how long the daemon
    drives the idle twist (`--idle`, zeros by default) before handing back to
    the gait, so the robot is not let go of on one foot. The owner runs it as a
    one-shot with `policy add ... --hold <seconds>`.

Before anything is uploaded, `publish` checks the graph is `[1,61] -> [1,14]`
(a 51-D legacy policy is refused with a message), runs it on plausible inputs
and refuses NaNs or a constant output, fills the `training` block from git and
wandb (task, commit, branch, dirty flag, run, checkpoint), and refuses to
overwrite an existing `.onnx` in the repo without `--force`. Repos are created
private; `--no-private` for public, `--tag v1` to tag the revision.

Only constant-command policies are publishable this way. Phase-driven moves
(the ground pick) and the posture-flag sit↔stand are driven by the daemon
itself and live in the official set, `pollen-robotics/microduck-policies`.

## Tests

```bash
uv run --with pytest pytest tests/
```

CPU-only config-invariant and reward-function regression tests — they lock in
joint-index mappings, reward sign conventions, and NaN guards.

## Related projects

- [microduck_rl](https://github.com/pollen-robotics/microduck_rl) — the upstream training foundation for Microduck Go

- [microduck](https://github.com/pollen-robotics/microduck) — the Microduck project home, including the onboard runtime that runs the exported policies
- [mjlab](https://github.com/mujocolab/mjlab) — the training framework (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/Rhoban/bam) — better actuator models, by Rhoban

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
3D model files are licensed under Creative Commons BY-SA-NC.
