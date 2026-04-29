# Truck Simulation MuJoCo

End-to-end MuJoCo truck-loading demo with YOLO-based vision, A* planning, a simple H1 posture controller, and MP4/report export.

## Project Structure

```text
.
├── bootstrap_venv.ps1      # Create the local virtual environment and install dependencies
├── env.py                  # MuJoCo scene wrapper and task state helpers
├── vision.py               # Day 2 vision pipeline using YOLO with a color fallback
├── planner.py              # Day 3 A* box-to-slot planner
├── executor.py             # Day 4 execution and rollout capture
├── pipeline.py             # Day 5 end-to-end orchestration and report export
├── h1_controller.py        # H1 posture / balance controller used during rollout
├── scene.xml               # Top-level MuJoCo scene
├── requirements.txt        # Python dependencies
├── outputs/                # Generated reports, images, and videos
├── mujoco_menagerie/       # Local Menagerie checkout for the H1 assets
└── assets/                 # Local asset junctions used by the scene
```

## Setup

From PowerShell in the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.ootstrap_venv.ps1
```

If you want to activate the environment manually after it is created:

```powershell
& ".\.venv-day5\Scripts\Activate.ps1"
```

## Run Commands

Run the full Day 5 pipeline:

```powershell
& ".\.venv-day5\Scripts\python.exe" pipeline.py --output-dir outputs\day5_run
```

If you activated the environment first, you can also use:

```powershell
python pipeline.py --output-dir outputs\day5_run
```

Run individual stages if you want to inspect them separately:

```powershell
python vision.py --output-dir outputs\day2
python planner.py --output outputs\day3\plan.json
python executor.py --plan outputs\day3\plan.json --output outputs\day4
```

To force Ultralytics to fetch YOLO weights automatically on the first run, add:

```powershell
--auto-download-weights
```

## Outputs

The pipeline writes its artifacts into the output directory you pass in:

- `vision/` for the top-down frame, annotated frame, and detection JSON
- `execution/` for rollout frames and execution JSON
- `demo.mp4` for the video playback
- `summary.png` for the visual summary
- `final_report.json` for the combined report

## Notes

- `outputs/`, virtual environments, local Menagerie assets, and downloaded YOLO weights are ignored by git.
- The H1 control layer keeps the robot upright during the rollout, but it is still a posture controller rather than full humanoid locomotion.