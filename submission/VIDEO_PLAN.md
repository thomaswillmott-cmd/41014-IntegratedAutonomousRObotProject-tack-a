# Five-minute demonstration plan

The prepared edit lasts **4:13.52**. Add your own voice over
`media/review_edit.mp4`; it is silent. Explain only claims you understand.
The media folder is local and intentionally excluded from GitHub.

## What to show

| Time | Visual | Talking points |
| --- | --- | --- |
| 0:00–0:20 | Title and contribution | Thomas Willmott, Track A, objective and accurate AI declaration. |
| 0:20–0:55 | Model, sensing and frames | Differential drive, state x/y/heading, encoders, camera correction and LiDAR mapping. |
| 0:55–1:33.84 | Complete nominal clip, 1x | Room, active route, camera/LiDAR, truth/estimate/uncertainty and visual docking. |
| 1:33.84–2:03.84 | EKF and controller | Prediction/correction, uncertainty, LQR weights and limits of the stability result. |
| 2:03.84–2:43.52 | Complete disturbed clip, 1x | Push occurs at video time about 2:13.84; forced blackout at about 2:15.84–2:21.84. Slower dead reckoning and eventual visual docking. |
| 2:43.52–3:18.52 | Three-seed results | Success, contacts, time, estimation/tracking error, final distance and heading. |
| 3:18.52–3:43.52 | Tracking graph | Initial-plan deviation versus active-path error; reference switches do not mean physical jumps. |
| 3:43.52–4:13.52 | Limits, reproducibility and sources | Known poses, ideal camera, restricted layouts, no full SLAM/hardware/charging; repeatable commands and declared sources. |

Rebuild after recording the two clips with `python submission/build_video_edit.py`.
Preserve/rename an existing edit first; the builder does not overwrite it.

## Narration draft

Use this as a guide, not a claim of unaided authorship or a script to memorise
without understanding. Shorten sentences to fit each section, or hold a frame
longer in your editor while keeping the total below five minutes.

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

**Tracking graph:** “The top panel measures deviation from the first generated
route. It includes deliberate route changes. The bottom panel measures distance
to the currently active reference. At the docking transition, that reference
changes to the station approach line, which can cause a jump without the robot
physically jumping. Neither graph measures EKF position estimation error.”

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
