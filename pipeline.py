"""
Day 5: end-to-end run that ties together vision, planning, execution,
metrics, and MP4 video export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import imageio.v2 as imageio
import numpy as np

from vision import TopDownVisionPipeline
from planner import AStarPlanner
from executor import SimulationExecutor
from env import TruckLoadingEnv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 5 end-to-end pipeline")
    parser.add_argument("--xml", default="scene.xml", help="Path to the MuJoCo scene XML")
    parser.add_argument("--weights", default=None, help="Path to YOLOv8-nano weights")
    parser.add_argument("--auto-download-weights", action="store_true", help="Allow Ultralytics to fetch weights")
    parser.add_argument("--output-dir", default="outputs/day5", help="Directory for outputs")
    parser.add_argument("--fps", type=int, default=8, help="Video FPS")
    parser.add_argument("--frame-repeat", type=int, default=2, help="Repeat each rendered frame this many times in the MP4")
    return parser


def write_video(frame_paths: List[str], video_path: Path, fps: int, frame_repeat: int) -> None:
    with imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            for _ in range(max(1, frame_repeat)):
                writer.append_data(frame)


def write_summary_image(output_dir: Path, vision_report: Dict[str, object], execution: Dict[str, object], metrics: Dict[str, object]) -> Path:
    topdown_path = Path(str(vision_report["annotated_file"]))
    first_frame = Path(str(execution["frame_paths"][0]))
    last_frame = Path(str(execution["frame_paths"][-1]))

    topdown = cv2.imread(str(topdown_path))
    first = cv2.imread(str(first_frame))
    last = cv2.imread(str(last_frame))

    if topdown is None or first is None or last is None:
        raise RuntimeError("Unable to load summary image inputs")

    topdown = cv2.resize(topdown, (640, 480), interpolation=cv2.INTER_CUBIC)
    first = cv2.resize(first, (640, 360), interpolation=cv2.INTER_CUBIC)
    last = cv2.resize(last, (640, 360), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((960, 1280, 3), 25, dtype=np.uint8)
    canvas[:480, :640] = topdown
    canvas[:360, 640:] = first
    canvas[480:840, 640:] = last

    text_panel = np.full((480, 640, 3), 35, dtype=np.uint8)
    lines = [
        "Project Summary",
        f"completion: {metrics['completion']:.1f}%",
        f"collisions: {metrics['collisions']}",
        f"distance: {metrics['distance']:.2f}",
        f"steps: {metrics['steps']}",
        f"priority_score: {metrics['priority_score']}",
        f"score: {metrics['score']}",
        f"vision: {vision_report['source']}",
        f"video frames: {len(execution['frame_paths'])}",
    ]
    y = 56
    for index, line in enumerate(lines):
        font_scale = 0.92 if index == 0 else 0.72
        thickness = 2 if index == 0 else 1
        cv2.putText(text_panel, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)
        y += 52 if index == 0 else 40

    canvas[480:, :640] = text_panel

    summary_path = output_dir / "summary.png"
    cv2.imwrite(str(summary_path), canvas)
    return summary_path


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = TruckLoadingEnv(xml_path=args.xml)
    env.reset()

    vision = TopDownVisionPipeline(
        env=env,
        weights_path=args.weights,
        auto_download_weights=args.auto_download_weights,
    )
    vision_dir = output_dir / "vision"
    vision_report = vision.run_once(output_dir=str(vision_dir))

    planner = AStarPlanner(env)
    plan = planner.plan()
    plan_path = output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    executor = SimulationExecutor(env)
    execution = executor.execute(plan, output_dir=str(output_dir / "execution"))

    video_path = output_dir / "demo.mp4"
    write_video(execution.frame_paths, video_path, fps=args.fps, frame_repeat=args.frame_repeat)
    summary_path = write_summary_image(output_dir, vision_report, execution.to_dict(), env.compute_metrics())

    final_report: Dict[str, object] = {
        "vision": vision_report,
        "plan": plan.to_dict(),
        "execution": execution.to_dict(),
        "metrics": env.compute_metrics(),
        "video_path": str(video_path),
        "summary_image": str(summary_path),
    }

    report_path = output_dir / "final_report.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()