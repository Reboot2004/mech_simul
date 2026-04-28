"""
Truck Loading Simulation Environment
Day 1: MuJoCo scene setup + state management
"""

import mujoco
import numpy as np
import json
import os
from pathlib import Path

# ─────────────────────────────────────────
# Box metadata: id → priority, name, color
# Priority 3 = highest (load deepest/first)
# Priority 1 = lowest  (load near door/last)
# ─────────────────────────────────────────
BOX_META = {
    "box_1": {"priority": 1, "color": "red",    "size": [0.12, 0.12, 0.1]},
    "box_2": {"priority": 2, "color": "orange", "size": [0.12, 0.12, 0.1]},
    "box_3": {"priority": 3, "color": "green",  "size": [0.12, 0.12, 0.1]},
    "box_4": {"priority": 2, "color": "orange", "size": [0.12, 0.12, 0.1]},
    "box_5": {"priority": 3, "color": "green",  "size": [0.12, 0.12, 0.1]},
}

# Truck slots: name → (x, y, z) world position, depth_rank
# depth_rank: higher = deeper inside truck = for high priority boxes
TRUCK_SLOTS = {
    "back_left":   {"pos": [1.05, -0.35, 0.15], "depth_rank": 3},
    "back_mid":    {"pos": [1.50, -0.35, 0.15], "depth_rank": 3},
    "back_right":  {"pos": [1.95, -0.35, 0.15], "depth_rank": 3},
    "front_left":  {"pos": [1.05,  0.25, 0.15], "depth_rank": 1},
    "front_mid":   {"pos": [1.50,  0.25, 0.15], "depth_rank": 1},
    "front_right": {"pos": [1.95,  0.25, 0.15], "depth_rank": 1},
}


class TruckLoadingEnv:
    def __init__(self, xml_path="scene.xml"):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.reset()

        print("✅ TruckLoadingEnv initialised")
        print(f"   MuJoCo version : {mujoco.__version__}")
        print(f"   Boxes          : {list(BOX_META.keys())}")
        print(f"   Truck slots    : {list(TRUCK_SLOTS.keys())}")

    def reset(self):
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # State tracking
        self.slot_occupancy  = {s: None for s in TRUCK_SLOTS}   # slot → box_id or None
        self.box_placement   = {b: None for b in BOX_META}      # box  → slot or None
        self.collision_count = 0
        self.total_distance  = 0.0
        self.action_log      = []
        self.step_count      = 0
        self.frames          = []   # for video
        return self.get_scene_state()

    # ── Low-level helpers ──────────────────────────────────────────

    def get_box_position(self, box_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, box_name)
        if body_id < 0:
            raise KeyError(f"Unknown body: {box_name}")
        return self.data.xpos[body_id].copy()

    def get_body_position(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise KeyError(f"Unknown body: {body_name}")
        return self.data.xpos[body_id].copy()

    def get_ee_position(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()

        for body_name in ("torso_link", "pelvis", "torso"):
            try:
                return self.get_body_position(body_name)
            except KeyError:
                continue
        raise RuntimeError("No reference position found for the robot")

    def render_camera(self, camera_name: str, width: int = 640, height: int = 480) -> np.ndarray:
        safe_width = min(width, 640)
        safe_height = min(height, 480)
        renderer = mujoco.Renderer(self.model, height=safe_height, width=safe_width)
        renderer.update_scene(self.data, camera=camera_name)
        frame = renderer.render()
        renderer.close()
        return frame

    def get_scene_state(self) -> dict:
        """Returns full state dict for the planner."""
        state = {
            "boxes": {},
            "slots": {},
            "ee_pos": self.get_ee_position().tolist(),
            "collisions": self.collision_count,
            "distance": round(self.total_distance, 3),
            "steps": self.step_count,
        }
        for box in BOX_META:
            pos = self.get_box_position(box)
            state["boxes"][box] = {
                "position":  pos.tolist(),
                "priority":  BOX_META[box]["priority"],
                "color":     BOX_META[box]["color"],
                "placed":    self.box_placement[box] is not None,
                "slot":      self.box_placement[box],
            }
        for slot, info in TRUCK_SLOTS.items():
            state["slots"][slot] = {
                "position":   info["pos"],
                "depth_rank": info["depth_rank"],
                "occupied":   self.slot_occupancy[slot] is not None,
                "box":        self.slot_occupancy[slot],
            }
        return state

    # ── High-level actions (used by planner) ──────────────────────

    def pick(self, box_name: str) -> bool:
        if box_name not in BOX_META:
            print(f"❌ pick: unknown box {box_name}")
            return False
        if self.box_placement[box_name] is not None:
            print(f"⚠️  pick: {box_name} already placed")
            return False

        ee_pos  = self.get_ee_position()
        box_pos = self.get_box_position(box_name)
        dist    = float(np.linalg.norm(ee_pos - box_pos))

        self.total_distance += dist
        self.step_count     += 1
        self.action_log.append(f"pick({box_name})")

        print(f"🤏 pick({box_name})  dist={dist:.3f}m")
        return True

    def move(self, slot_name: str) -> bool:
        if slot_name not in TRUCK_SLOTS:
            print(f"❌ move: unknown slot {slot_name}")
            return False
        if self.slot_occupancy[slot_name] is not None:
            print(f"⚠️  move: slot {slot_name} already occupied")
            return False

        ee_pos   = self.get_ee_position()
        slot_pos = np.array(TRUCK_SLOTS[slot_name]["pos"])
        dist     = float(np.linalg.norm(ee_pos - slot_pos))

        self.total_distance += dist
        self.step_count     += 1
        self.action_log.append(f"move({slot_name})")

        print(f"🚀 move({slot_name})  dist={dist:.3f}m")
        return True

    def place(self, box_name: str, slot_name: str) -> bool:
        if box_name not in BOX_META:
            print(f"❌ place: unknown box {box_name}")
            return False
        if self.box_placement[box_name] is not None:
            print(f"⚠️  place: {box_name} already placed")
            return False
        if self.slot_occupancy[slot_name] is not None:
            self.collision_count += 1
            print(f"💥 place COLLISION: slot {slot_name} occupied!")
            return False

        # Teleport box to slot position (kinematic placement)
        body_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, box_name)
        slot_pos = np.array(TRUCK_SLOTS[slot_name]["pos"])

        # Find the joint for this free body and set its position
        jnt_id   = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[jnt_id]
        self.data.qpos[qpos_adr:qpos_adr+3] = slot_pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]  # upright quaternion

        mujoco.mj_forward(self.model, self.data)

        # Update state
        self.slot_occupancy[slot_name]  = box_name
        self.box_placement[box_name]    = slot_name
        self.step_count                += 1
        self.action_log.append(f"place({box_name})")

        print(f"📦 place({box_name}) → {slot_name}")
        return True

    # ── Metrics ───────────────────────────────────────────────────

    def compute_metrics(self) -> dict:
        total   = len(BOX_META)
        placed  = sum(1 for v in self.box_placement.values() if v is not None)
        completion = round(placed / total * 100, 1)

        # Priority placement score: high priority boxes should be in deep slots
        priority_score = 0
        for box, slot in self.box_placement.items():
            if slot is None:
                continue
            box_priority  = BOX_META[box]["priority"]        # 1-3
            slot_depth    = TRUCK_SLOTS[slot]["depth_rank"]  # 1 or 3
            # Perfect match = priority 3 in depth 3, priority 1 in depth 1
            if box_priority == slot_depth:
                priority_score += 1
            elif abs(box_priority - slot_depth) == 2:
                priority_score -= 1  # penalty for wrong placement

        score = (
            completion * 0.4
            + max(0, 1 - self.collision_count / 10) * 100 * 0.3
            + max(0, 1 - self.total_distance / 20)  * 100 * 0.2
            + max(0, 1 - self.step_count / 50)      * 100 * 0.1
        )

        return {
            "completion":      completion,
            "collisions":      self.collision_count,
            "distance":        round(self.total_distance, 2),
            "steps":           self.step_count,
            "priority_score":  priority_score,
            "score":           round(score, 1),
        }

    def get_output(self) -> dict:
        return {
            "state": self.get_scene_state(),
            "actions": self.action_log,
            "metrics": self.compute_metrics(),
        }

    def save_output(self, path="output.json"):
        out = self.get_output()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n📊 Output saved → {output_path}")
        print(json.dumps(out, indent=2))

    def render_topdown(self, width=640, height=480) -> np.ndarray:
        """Render top-down frame for YOLO (Day 2)."""
        return self.render_camera("topdown_cam", width=width, height=height)

    def step_sim(self, n=1, controller=None):
        for _ in range(n):
            if controller is not None:
                controller.apply()
            mujoco.mj_step(self.model, self.data)


# ─────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────
if __name__ == "__main__":
    env = TruckLoadingEnv(xml_path="scene.xml")

    print("\n── Initial scene state ──")
    state = env.get_scene_state()
    for box, info in state["boxes"].items():
        print(f"  {box}: priority={info['priority']} pos={[round(p,2) for p in info['position']]}")

    print("\n── Slot positions ──")
    for slot, info in state["slots"].items():
        print(f"  {slot}: depth_rank={info['depth_rank']} pos={info['position']}")

    print("\n✅ Day 1 complete — scene loaded, state readable")
