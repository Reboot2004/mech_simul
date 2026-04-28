"""
Day 3: A* planner for the truck loading task.

The planner searches over box-to-slot assignments using a cost function that
combines travel distance, priority mismatch, and a simple collision-risk term.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from env import BOX_META, TRUCK_SLOTS, TruckLoadingEnv


def _rounded_position(values: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(round(float(value), 3) for value in values)  # type: ignore[return-value]


@dataclass(frozen=True)
class PlannerState:
    current_pos: Tuple[float, float, float]
    remaining_boxes: Tuple[str, ...]
    free_slots: Tuple[str, ...]


@dataclass
class PlannedStep:
    box_id: str
    slot_id: str
    travel_dist: float
    priority_penalty: float
    collision_risk: float
    step_cost: float


@dataclass
class PlannerResult:
    steps: List[PlannedStep]
    total_cost: float
    expanded_nodes: int
    goal_reached: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "steps": [asdict(step) for step in self.steps],
            "total_cost": round(self.total_cost, 3),
            "expanded_nodes": self.expanded_nodes,
            "goal_reached": self.goal_reached,
        }


class AStarPlanner:
    def __init__(self, env: TruckLoadingEnv):
        self.env = env

    def _box_position(self, box_id: str) -> np.ndarray:
        return np.array(self.env.get_box_position(box_id), dtype=float)

    def _slot_position(self, slot_id: str) -> np.ndarray:
        return np.array(TRUCK_SLOTS[slot_id]["pos"], dtype=float)

    def _box_priority(self, box_id: str) -> int:
        return int(BOX_META[box_id]["priority"])

    def _slot_depth(self, slot_id: str) -> int:
        return int(TRUCK_SLOTS[slot_id]["depth_rank"])

    def _frontier_boxes(self, remaining_boxes: Iterable[str]) -> List[str]:
        remaining = list(remaining_boxes)
        if not remaining:
            return []
        max_priority = max(self._box_priority(box_id) for box_id in remaining)
        return [box_id for box_id in remaining if self._box_priority(box_id) == max_priority]

    def _priority_penalty(self, box_id: str, slot_id: str) -> float:
        return 0.0 if self._box_priority(box_id) == self._slot_depth(slot_id) else 10.0

    def _collision_risk(self, state: PlannerState, slot_id: str) -> float:
        slot_pos = self._slot_position(slot_id)
        occupied_slots = [slot for slot in TRUCK_SLOTS if slot not in state.free_slots]

        risk = 0.0
        for other_slot in occupied_slots:
            other_pos = self._slot_position(other_slot)
            delta = np.abs(slot_pos - other_pos)
            if delta[0] <= 0.55 and delta[1] <= 0.45:
                risk += 2.0

        if self._slot_depth(slot_id) == 1:
            risk += 1.0
        return risk

    def _step_cost(self, state: PlannerState, box_id: str, slot_id: str) -> Tuple[float, float, float, float]:
        current_pos = np.array(state.current_pos, dtype=float)
        box_pos = self._box_position(box_id)
        slot_pos = self._slot_position(slot_id)

        travel_dist = float(np.linalg.norm(current_pos - box_pos) + np.linalg.norm(box_pos - slot_pos))
        priority_penalty = self._priority_penalty(box_id, slot_id)
        collision_risk = self._collision_risk(state, slot_id)
        step_cost = travel_dist + priority_penalty + collision_risk
        return travel_dist, priority_penalty, collision_risk, step_cost

    def _heuristic(self, state: PlannerState) -> float:
        if not state.remaining_boxes:
            return 0.0

        current_pos = np.array(state.current_pos, dtype=float)
        free_slots = list(state.free_slots)
        if not free_slots:
            return 0.0

        frontier = self._frontier_boxes(state.remaining_boxes)
        box_pos = self._box_position(frontier[0])
        slot_pos = np.array([self._slot_position(slot_id) for slot_id in free_slots], dtype=float)
        estimated = float(np.min(np.linalg.norm(slot_pos - box_pos, axis=1)) + np.linalg.norm(current_pos - box_pos))
        return estimated * max(0.25, len(state.remaining_boxes) * 0.15)

    def _sort_slots_for_box(self, box_id: str, state: PlannerState) -> List[str]:
        box_priority = self._box_priority(box_id)
        slots = list(state.free_slots)

        def slot_key(slot_id: str) -> Tuple[float, float, float]:
            depth_gap = abs(box_priority - self._slot_depth(slot_id))
            slot_pos = self._slot_position(slot_id)
            current_pos = np.array(state.current_pos, dtype=float)
            travel_hint = float(np.linalg.norm(current_pos - slot_pos))
            return depth_gap, travel_hint, slot_pos[0]

        return sorted(slots, key=slot_key)

    def plan(self) -> PlannerResult:
        start_state = PlannerState(
            current_pos=_rounded_position(self.env.get_ee_position()),
            remaining_boxes=tuple(box_id for box_id, slot in self.env.box_placement.items() if slot is None),
            free_slots=tuple(slot_id for slot_id, box in self.env.slot_occupancy.items() if box is None),
        )

        open_heap: List[Tuple[float, int, float, PlannerState]] = []
        counter = 0
        start_h = self._heuristic(start_state)
        heapq.heappush(open_heap, (start_h, counter, 0.0, start_state))

        best_costs: Dict[PlannerState, float] = {start_state: 0.0}
        parents: Dict[PlannerState, Tuple[Optional[PlannerState], Optional[PlannedStep]]] = {
            start_state: (None, None)
        }

        expanded_nodes = 0
        goal_state: Optional[PlannerState] = None

        while open_heap:
            _, _, g_cost, state = heapq.heappop(open_heap)
            if g_cost > best_costs.get(state, float("inf")):
                continue

            expanded_nodes += 1
            if not state.remaining_boxes:
                goal_state = state
                break

            frontier_boxes = self._frontier_boxes(state.remaining_boxes)
            for box_id in frontier_boxes:
                for slot_id in self._sort_slots_for_box(box_id, state):
                    travel_dist, priority_penalty, collision_risk, step_cost = self._step_cost(state, box_id, slot_id)
                    next_state = PlannerState(
                        current_pos=_rounded_position(self._slot_position(slot_id)),
                        remaining_boxes=tuple(item for item in state.remaining_boxes if item != box_id),
                        free_slots=tuple(item for item in state.free_slots if item != slot_id),
                    )
                    next_g = g_cost + step_cost
                    if next_g >= best_costs.get(next_state, float("inf")):
                        continue

                    best_costs[next_state] = next_g
                    parents[next_state] = (
                        state,
                        PlannedStep(
                            box_id=box_id,
                            slot_id=slot_id,
                            travel_dist=round(travel_dist, 3),
                            priority_penalty=round(priority_penalty, 3),
                            collision_risk=round(collision_risk, 3),
                            step_cost=round(step_cost, 3),
                        ),
                    )
                    counter += 1
                    heapq.heappush(open_heap, (next_g + self._heuristic(next_state), counter, next_g, next_state))

        if goal_state is None:
            return PlannerResult(steps=[], total_cost=0.0, expanded_nodes=expanded_nodes, goal_reached=False)

        steps: List[PlannedStep] = []
        cursor = goal_state
        while True:
            parent_state, step = parents[cursor]
            if parent_state is None or step is None:
                break
            steps.append(step)
            cursor = parent_state
        steps.reverse()

        return PlannerResult(
            steps=steps,
            total_cost=best_costs[goal_state],
            expanded_nodes=expanded_nodes,
            goal_reached=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 3 A* planner")
    parser.add_argument("--xml", default="scene.xml", help="Path to the MuJoCo scene XML")
    parser.add_argument("--output", default="outputs/day3/plan.json", help="Path for the plan JSON")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    env = TruckLoadingEnv(xml_path=args.xml)
    planner = AStarPlanner(env)
    result = planner.plan()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()