# UTS 41014: TurtleBot3 Visual Waypoint Navigation

**Assessment 4, Track A — simulation in MuJoCo**

Navigate an obstacle-filled room using 2D LiDAR, estimate robot pose from wheel
encoders and an ArUco camera, and visually dock at a charging-station target.
Navigation and control use the **EKF estimate**, never the simulator's true pose.

| Submission details | Details |
| --- | --- |
| Project label | 1412 (self-selected; no official group number assigned) |
| Student | Thomas Willmott — SID 25494826; individual implementation |
| GitHub repository URL | [41014-IntegratedAutonomousRObotProject-tack-a](https://github.com/thomaswillmott-cmd/41014-IntegratedAutonomousRObotProject-tack-a) |
| Five-minute demo video URL | TODO |
| Student contribution | Defined/refined project behaviour, ran simulation trials, reviewed results and requested corrections; video/submission preparation. Implementation and documentation were developed with the AI assistance declared below. |

## Install and run

Windows, Python 3.12 x64; tested with Python 3.12.10. Open a terminal in this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run one random layout; change the seed to select another reproducible layout:

```powershell
.\.venv\Scripts\python.exe run_robot.py --layout random --seed 22 --output results/seed22
```

Run three randomly selected seeds consecutively, or repeat the reported batch:

```powershell
.\.venv\Scripts\python.exe run_batch.py
.\.venv\Scripts\python.exe run_batch.py --seeds 524882 407433 547246
```

All injected errors are enabled by default. Add `--headless` for faster execution
without windows (camera rendering still needs OpenGL), or `--clean` to a single
run for a comparison without injections. No ROS or physical webcam is required.

Batches advance on completion or at 180 simulated seconds and save to timestamped
folders. Single runs save on exit; without `--output`, they overwrite `results/latest`.
P pauses, R resets/clears the log, M toggles manual mode, and dashboard Esc exits.

## A. Platform model

State `q=[x,y,theta]`, input `u=[v,omega]`, wheel radius `r=0.033m`, axle separation
`L=0.288m`. The planar differential-drive model is:

```text
x_dot = v*cos(theta)        v = r*(wheel_rate_right + wheel_rate_left)/2
y_dot = v*sin(theta)    omega = r*(wheel_rate_right - wheel_rate_left)/L
theta_dot = omega
```

The local reference-frame linearisation uses `dt=0.02s`,
`A=[[1,0,0],[0,1,v_ref*dt],[0,0,1]]` and
`B=[[dt,0],[0,0],[0,dt]]`. Controllability rank is 3 at nonzero forward speed
and 2 at rest. Camera range/bearing alone have instantaneous measurement rank 2;
including marker-derived yaw gives rank 3 away from singular geometry. Without
camera observations, encoders alone do not establish absolute world pose.
See [model.py](track_a/model.py) for the Jacobians and rank-analysis functions.

The model assumes a flat floor and nominal no-slip rolling. MuJoCo separately
integrates chassis/wheel dynamics and contacts, which can violate these assumptions.

## B. Sensing, calibration and frames

```text
World (x/y floor, z up)
  ├─ Robot axle frame (x forward, y left, z up)
  │    ├─ LiDAR: robot translation [-0.049, 0, 0.19] m
  │    └─ Camera: robot translation [0.11, 0, 0.23] m
  │              OpenCV axes: x right, y down, z forward
  └─ Known marker centre: [2.82, 1.4, 0.25] m; black square width 0.30 m
```

LiDAR populates the occupancy map using estimated pose. ArUco detection and PnP
produce camera range, bearing and yaw, accounting for mount and marker orientation.
Camera observations correct the EKF; LiDAR provides mapping/clearance, not localisation.

[camera.json](calibration/camera.json): 100 calibration views, reprojection RMS
**0.037px**, held-out range RMSE **7.44mm** across 34 detections from 36 validation
poses. Zero distortion is constrained for the ideal pinhole renderer; these are
simulation results, not physical-camera calibration.

## C. Estimation

The 50Hz EKF predicts from wheel-angle increments using midpoint integration.
Process covariance includes wheel-distance variance `(0.02*abs(d_wheel))²+1e-9`
and a rotated body-frame model-error term:

```text
Q_slip = abs(ds)*diag(0.0009,0.0036,0.0006)
       + abs(dtheta)*diag(0.00001,0.00001,0.0009)
```

Camera updates use range standard deviation `0.015+0.015*range²` metres,
bearing standard deviation `1.5°`, and yaw standard deviation `5+2*range` degrees.
These conservative engineering settings exceed ideal calibration errors; they are
not identified physical noise. Wrapped innovations, NIS gating (11.34 for three
measurements) and Joseph covariance updates handle outliers/numerics.
See [estimation.py](track_a/estimation.py).

## D. Planning and control

Heading-aware A* replans on a **2cm LiDAR occupancy grid**, treating unknown cells
as provisionally free. Default cost is `distance/0.18 + turn_angle/0.9`, a travel-time
surrogate. Planning radius is `0.16m + 0.5*major_position_sigma + 0.02/sqrt(2)`;
live swept-body LiDAR checks add 1.5cm local padding without changing the real body.

Navigation uses scheduled discrete LQR with `Q=diag(6,14,3)`, `R=diag(2,1)`.
Lateral weighting discourages cross-track departures; input penalties moderate
corrections. At `v_ref=0.18m/s`, gain is approximately
`K=[[1.7023,0,0],[0,3.6645,2.0552]]`. Local closed-loop eigenvalues are
`[0.96595,0.96686,0.99204]`. This supports local stability, excluding saturation,
replanning and hybrid switching. Gains are engineering choices, not proven optimal.
Encoder wheel-speed feedback compensates drive bias.

Final docking uses separate pose feedback: distance gain 0.65, bearing gain 2,
final-orientation term -0.65, speed limited to 0.10m/s and turn rate to 0.65rad/s.
It requires fresh camera corrections and verifies alignment before declaring
completion. Full laws and limits: [navigation.py](track_a/navigation.py).

## E. Results and robustness

Default injected errors are encoder noise `σ=0.002rad`, LiDAR noise `σ=0.015m`,
corner noise `σ=0.65px`, wheel-drive multipliers `[0.97,1.02]`, camera blackout at
12–18s plus 25% random frame losses outside that interval, and a 0.8N sideways push
at 10–10.5s. Physical contact/slip comes from MuJoCo. Process covariance models
uncertainty; it is not an injected physical disturbance.

Without fresh camera corrections, navigation continues at 75% speed using encoder
prediction and live LiDAR checks; uncertainty can grow. Final docking waits for
vision. Confirmed sensor inconsistencies can trigger separate relocalisation.

Measured three-seed batch with all injected errors enabled:

| Seed | Docked | Time (s) | Mean XY estimation error (cm) | Active-path tracking RMSE (cm) | Final dock distance (cm) | Heading error (°) | Contact steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 524882 | Yes | 39.40 | 0.96 | 3.71 | 4.87 | 1.59 | 0 |
| 407433 | Yes | 39.04 | 1.03 | 3.43 | 4.82 | 1.30 | 0 |
| 547246 | Yes | 46.68 | 1.42 | 4.75 | 4.93 | 0.30 | 0 |

Evidence: [comparison.csv](results/submission_evidence/comparison.csv) and
[recorded metrics](results/submission_evidence/all_results.json). Three successes
do not establish reliability for arbitrary layouts. Tracking includes control lag
and reference switches, explaining why it can exceed estimation error.

Offline success requires controller completion, true final distance below 8cm,
heading error below 0.12rad, and zero logged obstacle contacts. Ground truth is
used only to generate simulated sensor data and for evaluation/display; it never
corrects the controller directly. Evaluation is separated in
[evaluation.py](track_a/evaluation.py) and [trial_report.py](track_a/trial_report.py).

Batches save reports, CSVs, telemetry, path history and trajectory/reference,
tracking, docking and estimation graphs. Estimation error is EKF-to-truth distance;
tracking error is truth-to-active-polyline distance. Deviation from the fixed first
plan includes legitimate replanning and excludes docking, since it ends at staging.
These are spatial distances, not time-synchronised trajectory errors.

Limitations: known initial/station poses, restricted random obstacle layouts,
ideal camera rendering, no full SLAM, no electrical charging model, and no hardware
validation. Gain trade-offs and broader failure rates need further matched trials.

## Verification and submission

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Last verification: **51 tests passed**, including in a fresh Python environment
installed from `requirements.txt` on Windows. The five-minute video must show task
completion, a varied/disturbed trial, estimate, uncertainty and truth. Add its URL
above and ensure markers can access the repository and video before submission.

Recording genuine 1080p clips (silent, 1x simulation playback):

```powershell
.\.venv\Scripts\python.exe record_demo.py --clean --output submission/media/nominal
.\.venv\Scripts\python.exe record_demo.py --output submission/media/disturbed
```

Each recording includes room, camera, LiDAR, truth, EKF estimate and uncertainty,
plus matching telemetry and assessment graphs. See [video plan](submission/VIDEO_PLAN.md)
for narration and the five-minute structure. Generated videos are excluded from Git;
host the finished video separately and link it here.

## Sources and AI declaration

Robot model/meshes: [ROBOTIS MuJoCo Menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie),
commit `d8344c0dbe7a00208d0301111523dde65efc174a`, Apache-2.0; included
[licence](models/robotis_tb3/LICENSE). Runtime scene assembly adapts collision,
contact and motor settings and adds sensors; original assets remain unchanged.
Methods use [MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html),
[OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) and
[SciPy's discrete Riccati solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html).

**Generative AI:** OpenAI ChatGPT/Codex supported explanations, code generation,
revision, debugging, tests, simulations and documentation. Record actual student
contributions separately; acknowledge any additional tutorial code/sources used.
