# Diagnosis: rpsv2c closed loop vs oracle demonstrations on the identical training layouts (2026-10-12)

## Scope and method

- **Policy traces.** rpsv2c closed loop on the 55 v2 training layouts, two runs (n_action_steps 50 and 10), giving
  110 episodes:
  - `reports/evaluation_evo1_rpsv2c_trainlayouts_v2scene_bf16_s8{,_n10}_position_visual.json`
  - `outputs/traces/`
- **Oracle reference.** The oracle demonstration of the same seed in `datasets/sim_so101_v2_all`:
  - episode = (task−1)·40 + rank of the seed among the successful generation seeds;
  - lengths match `reports/sim_v2_report_task_XX.json` exactly.
- **Grasp point.** Forward kinematics (MuJoCo) of the recorded joint states: gripperframe site + GripperIK offset, as
  the oracle uses.
- **Oracle phases.** approach → descend (last 4 cm above the grasp height) → close (gripper-closing onset + 20 steps)
  → carry. Only the first grasp is segmented.
- **Alignment and divergence.** Each policy step is matched to the nearest oracle grasp point under a forward-progress
  constraint. Divergence = the distance to the oracle path stays > 2 cm for 15 steps.
- **Tip replay.** The 27 pot-tip episodes were replayed open loop with the recorded actions from the same seed; the
  physics is deterministic and all 27 tips reproduced. Every robot–pot contact was logged.
- Script: `scripts/diag_policy_vs_oracle.py`. Data: `reports/diag_policy_vs_oracle.json`.

## 1. Where the policy leaves the oracle

| phase of first divergence (> 2 cm for 0.5 s) | episodes |
|---|---:|
| approach | 48 |
| descend | 8 |
| close | 2 |
| carry (place / transport; mostly tasks 02, 07, 10, 11) | 40 |
| no divergence | 12 |

- Median deviation over an episode: 4.3 cm. At the onset of divergence: 2.25 cm.
- Joint deviation at onset (|Δ| median): elbow 8.0°, shoulder_lift 6.7°, wrist_roll 7.0°, wrist_flex 4.5°, pan 3.0°.
- **Half of the episodes are already off the oracle path during the approach.** The error does not first appear at
  contact.

## 2. Gripper closing: right height, wrong place

| | policy (71 closings) | oracle |
|---|---:|---:|
| horizontal offset grasp point → target at closing onset | **3.1 cm** (per task 1.7–4.1) | 0.06 cm |
| vertical offset | 0.85 cm | 0.78 cm |

- The timing / height of closing is correct. The **lateral position is off by about 3 cm**, more than a 25 mm cube.
  Grasps miss, and the policy then continues the place motion with an empty gripper.

## 3. Pot tips (27 episodes; tasks 01, 03, 04, 05, 06)

- **Tasks 05 / 06** (cube inside the pot; 20 / 20 episodes tipped):
  - The tip happens 0.1–0.7 s after the first contact, while descending into the pot, before closing. The gripper is
    open (about 25) and not grasping (only 3 / 27 tips with a blocked closing gripper).
  - The grasp point is inside the opening, **0.9–2.9 cm from the pot axis (oracle about 0.0–0.1 cm)**, at z 2–5 cm
    (rim 5.5–7 cm).
  - The gripper is almost vertical (tilt 3–5°). An earlier wrist_flex hypothesis was rejected: elbow / lift compensate
    it (joint redundancy).
- **Contacts at the tip** (replay):

  | robot part touching the pot | episodes |
  |---|---:|
  | jaws (gripper body / moving jaw) | 24 / 25 of 27 |
  | wrist | 12 |
  | wrist-camera mount / camera | 11 / 7 |
  | arm links | 1–2 |

- **Orientation:** |Δ wrist_roll| at the tip is 15.7° (median) vs the oracle. The oracle aligns the jaw axis with an
  exact yaw.
  - In the small cup (inner Ø 8.4 cm) and the pots, a 1–3 cm offset plus about 16° roll brings the open jaw and the
    camera mount into the wall.
- **Tasks 01 / 03:**
  - Tips come after a missed grasp: the empty, open gripper (about 35) is lowered into the pot to z 2.4–3.5 cm.
  - The oracle releases at rim + 2 cm (z ≈ 8 cm). The policy over-descends by about 5 cm.
- **Size of the deviation just before the tip (medians):**
  - grasp point 4.9 cm from the oracle path;
  - 1–3 cm lateral offset from the pot axis (oracle about 0);
  - wrist_roll about 16°;
  - signed joint deltas: lift +4.7°, elbow +6.7°, wrist_flex −14.6°. These largely cancel in the gripper pose (tilt
    3–5°).

## Implications for the v3 data

1. The precision problem starts in the approach (lateral offset 2–4 cm). The demos give no example of *correcting*
   such an offset; oracle trajectories are always on the path.
   - → Recovery demonstrations: inject a perturbation (±2–4 cm, ±15° yaw) during the oracle approach and record its
     closed-loop correction (DART / DAgger-style). Keep only successful episodes.
2. Container entry is the most fragile phase (tasks 05 / 06: 100 % tips). The jaw orientation (wrist_roll) and
   centring decide wall contact.
   - → Over-sample in-container grasps (05 / 06) with wide yaw, and a pre-grasp hover above the container axis before
     descending, so the policy sees a "centre first, then descend" pattern.
3. Place height after a grasp (tasks 01 / 03): the policy descends too far.
   - → Place demos with varied but explicit release heights; the release cue is the gripper opening above the rim.
- Position randomization (`position_wide`) addresses unseen positions. It does not address points 1–3 on its own.
