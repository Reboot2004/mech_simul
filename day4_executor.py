"""
Day 4: execute the planned pick-move-place sequence in MuJoCo.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from day3_planner import AStarPlanner, PlannedStep, PlannerResult
from h1_controller import H1TaskController
from env import TruckLoadingEnv


@dataclass
class ExecutionStep:
    index: int
    box_id: str
    slot_id: str
    pick_ok: bool
    move_ok: bool
    place_ok: bool
    collision_count: int
    distance: float


@dataclass
class ExecutionResult:
    success: bool
    steps: List[ExecutionStep]
    frame_paths: List[str]
    metrics: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "steps": [asdict(step) for step in self.steps],
            "frame_paths": self.frame_paths,
            "metrics": self.metrics,
        }


class SimulationExecutor:
    def __init__(self, env: TruckLoadingEnv, camera_name: str = "side_cam", frame_size: tuple[int, int] = (960, 720)):
        self.env = env
        self.camera_name = camera_name
        self.frame_width, self.frame_height = frame_size
        self.controller = H1TaskController(env.model, env.data)
        self.hold_steps = 1

    def _render_panel(self, width: int, height: int, lines: List[str]) -> np.ndarray:
        panel = np.full((height, width, 3), 28, dtype=np.uint8)
        y = 42
        for line in lines:
            cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            y += 36
        return panel

    def _capture_frame(self, action_text: str, step_index: int) -> object:
        topdown = self.env.render_camera("topdown_cam", width=640, height=480)
        topdown = cv2.resize(topdown, (960, 720), interpolation=cv2.INTER_CUBIC)

        side_view = self.env.render_camera(self.camera_name, width=640, height=480)
        side_view = cv2.resize(side_view, (320, 240), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((720, 1280, 3), 24, dtype=np.uint8)
        canvas[:, :960] = topdown
        canvas[24:264, 960:1280] = side_view if side_view.shape[1] == 320 else cv2.resize(side_view, (320, 240))

        header = [
            f"Step {step_index}",
            action_text,
            f"collisions: {self.env.collision_count}",
            f"distance: {self.env.total_distance:.2f}",
            f"completion: {self.env.compute_metrics()['completion']:.1f}%",
        ]
        panel = self._render_panel(320, 456, header)
        canvas[288:720, 960:1280] = panel[:432, :]

        cv2.rectangle(canvas, (0, 0), (959, 719), (255, 255, 255), 2)
        cv2.rectangle(canvas, (960, 0), (1279, 719), (90, 90, 90), 2)
        cv2.putText(canvas, "Top-down rollout", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Live status", (992, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def execute(self, plan: PlannerResult, output_dir: str = "outputs/day4") -> ExecutionResult:
        output_path = Path(output_dir)
        frame_dir = output_path / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        self.env.reset()
        self.controller.reset()

        steps: List[ExecutionStep] = []
        frame_paths: List[str] = []
        frame_index = 0

        def save_frame(action_text: str) -> None:
            nonlocal frame_index
            frame = self._capture_frame(action_text, frame_index)
            frame_path = frame_dir / f"frame_{frame_index:04d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(str(frame_path))
            frame_index += 1

        self.controller.set_mode("stand")
        self.controller.step(24)
        save_frame("initial state")
        for _ in range(self.hold_steps):
            self.controller.step(1)
            save_frame("standing by")

        success = True
        for index, step in enumerate(plan.steps, start=1):
            self.controller.set_mode("pick", box_id=step.box_id, slot_id=step.slot_id)
            self.controller.step(18)
            pick_ok = self.env.pick(step.box_id)
            save_frame(f"pick({step.box_id})")
            for _ in range(self.hold_steps):
                self.controller.step(1)
                save_frame(f"holding {step.box_id}")

            move_ok = False
            place_ok = False
            if pick_ok:
                self.controller.set_mode("move", box_id=step.box_id, slot_id=step.slot_id)
                self.controller.step(18)
                move_ok = self.env.move(step.slot_id)
                save_frame(f"move({step.slot_id})")
                for _ in range(self.hold_steps):
                    self.controller.step(1)
                    save_frame(f"holding {step.slot_id}")

            if pick_ok and move_ok:
                self.controller.set_mode("place", box_id=step.box_id, slot_id=step.slot_id)
                self.controller.step(18)
                place_ok = self.env.place(step.box_id, step.slot_id)
                save_frame(f"place({step.box_id}, {step.slot_id})")
                for _ in range(self.hold_steps):
                    self.controller.step(1)
                    save_frame(f"placed {step.box_id}")
            else:
                save_frame(f"skipped place({step.box_id}, {step.slot_id})")

            steps.append(
                ExecutionStep(
                    index=index,
                    box_id=step.box_id,
                    slot_id=step.slot_id,
                    pick_ok=pick_ok,
                    move_ok=move_ok,
                    place_ok=place_ok,
                    collision_count=self.env.collision_count,
                    distance=round(self.env.total_distance, 3),
                )
            )

            if not (pick_ok and move_ok and place_ok):
                success = False
                break

        self.controller.set_mode("stand")
        self.controller.step(18)
        save_frame("final state")

        metrics = self.env.compute_metrics()
        result = ExecutionResult(
            success=success,
            steps=steps,
            frame_paths=frame_paths,
            metrics=metrics,
        )

        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "execution.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 4 simulation executor")
    parser.add_argument("--xml", default="scene.xml", help="Path to the MuJoCo scene XML")
    parser.add_argument("--plan", default="outputs/day3/plan.json", help="Plan JSON to execute")
    parser.add_argument("--output", default="outputs/day4", help="Output directory")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    env = TruckLoadingEnv(xml_path=args.xml)
    planner = AStarPlanner(env)

    plan_path = Path(args.plan)
    if plan_path.exists():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        plan = PlannerResult(
            steps=[PlannedStep(**step) for step in plan_data.get("steps", [])],
            total_cost=float(plan_data.get("total_cost", 0.0)),
            expanded_nodes=int(plan_data.get("expanded_nodes", 0)),
            goal_reached=bool(plan_data.get("goal_reached", False)),
        )
    else:
        plan = planner.plan()

    executor = SimulationExecutor(env)
    result = executor.execute(plan, output_dir=args.output)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()