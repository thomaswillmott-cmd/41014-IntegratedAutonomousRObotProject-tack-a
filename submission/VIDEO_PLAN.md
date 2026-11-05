# Five-minute demonstration plan

Aim for **4:30–4:50**, leaving room for natural pauses. Record your own voice;
the generated simulation clips are silent. Explain only claims you understand.
The media folder is local and intentionally excluded from GitHub.

## What to show

| Time | Visual | Talking points |
| --- | --- | --- |
| 0:00–0:25 | Project title and README task/pipeline | Thomas Willmott, Track A, TurtleBot3 in MuJoCo; objective is obstacle navigation and visual docking. Individual implementation; explain AI assistance accurately. |
| 0:25–1:00 | README model and frame diagram | Differential drive, state x/y/heading, wheel encoders, camera corrections, LiDAR mapping. Controller receives estimated state. |
| 1:00–1:45 | `media/nominal_seed22/simulation.mp4` | Genuine clean comparison. Explain room, active path, camera/LiDAR, actual and estimated pose, and numeric uncertainty. Play the complete clip at 1x; use remaining time to hold its final frame. |
| 1:45–2:15 | README EKF and controller section | Prediction versus correction, covariance, chosen Q/R and LQR weights. State why gain choices are engineering settings. |
| 2:15–3:00 | `media/disturbed_seed22/simulation.mp4` | All injections, same layout/seed. Highlight push at 10s, camera blackout 12–18s, slower dead reckoning, restored corrections and visual docking. Play complete clip at 1x; hold its final frame as needed. |
| 3:00–3:45 | Saved batch comparison and tracking/docking graphs | Success counts, completion time, estimation error versus tracking error, final dock distance/heading. Explain reference-switch spikes and uncertainty growth. |
| 3:45–4:20 | Limits and failure discussion | Restricted layouts, known starting/station poses, ideal camera, conservative/tuned covariances, no physical charging. Earlier failures are useful development evidence; do not present them as results from the final version. |
| 4:20–4:45 | Run commands, sources and contribution statement | Repeatable seeds, batch command, declared AI assistance and ROBOTIS assets; point to repository/video links. |

## Narration draft

Use this as a guide, not a claim of unaided authorship or a script to memorise
without understanding. Adjust timing to your speaking pace.

**Opening:** “This is my Track A integrated sensors and control project. It uses
a TurtleBot3 model in MuJoCo to navigate an obstacle-filled room and dock near an
ArUco-marked station. I developed it with ChatGPT and Codex assistance, including
generated code and debugging. My work included deciding the required behaviour,
testing the simulation and reviewing its results.”

**Pipeline:** “The state contains world x, y and heading. Wheel rotation predicts
forward movement and turning using differential-drive kinematics. The camera
detects the marker and produces range, bearing and heading measurements. These
correct the EKF estimate. LiDAR builds the obstacle map and checks clearance;
it does not directly correct absolute position in this implementation. A-star
plans through the map and the feedback controller follows that route using the
estimated state. Ground truth is kept separate for simulation and evaluation.”

**Nominal run:** “This first run disables injected errors but retains MuJoCo
contact physics and image detection. The left view shows the actual robot and
active path. The dashboard shows the camera, LiDAR and map. The bottom values
compare actual and estimated pose and show one-standard-deviation uncertainty.
The final approach uses camera-gated pose feedback, so docking needs a fresh
visual correction.”

**Design:** “The EKF predicts at fifty hertz. Its process covariance allows for
wheel uncertainty and unmodelled motion. Camera measurement covariance increases
with distance, and innovation gating rejects inconsistent observations. The LQR
weights penalise lateral error and heading error while limiting aggressive input.
Its local closed-loop eigenvalues are inside the unit circle, but that is not a
proof for the whole system with saturation, obstacle stops and replanning.”

**Disturbed run:** “The second run uses the same seed with encoder, LiDAR and
camera noise, unequal wheel drive, a sideways push and camera losses. Watch the
push at ten seconds and the camera blackout between twelve and eighteen seconds.
During missing camera corrections, the robot continues more slowly using encoder
prediction and live LiDAR checks. Uncertainty can grow because wheel odometry
cannot independently correct accumulated drift. Fresh camera measurements resume
the normal EKF updates.”

**Results:** “Across the recorded three-seed batch, all three trials docked
without recorded obstacle contacts. Mean position estimation error ranged from
zero point nine six to one point four two centimetres. This is different from
tracking error, which measures distance from the actual robot to the active path.
The first path can change when obstacles are discovered, so deviation from the
initial plan does not necessarily mean poor control. Final docking distance and
heading are reported separately.”

**Limits and conclusion:** “These results cover restricted generated layouts,
not arbitrary rooms. The initial pose and station pose are known, the camera is
an ideal renderer, and electrical charging is not modelled. The model and gain
choices still need broader comparisons before claiming optimality or guaranteed
safety. The repository provides installation instructions, individual-seed and
batch runs, numeric evidence, sources and an AI declaration.”

## Before submitting

- Replace the README video placeholder with the actual video URL.
- Verify both links open for the marker; a private repository needs explicit access.
- Show estimate **and uncertainty** alongside truth, including during disturbances.
- Keep final runtime under five minutes; do not hide unsuccessful trials in reported studies.
- Check the subject's official deadline, individual-submission arrangements and viva details.
- Submit the links on Canvas yourself after reviewing the final materials.
