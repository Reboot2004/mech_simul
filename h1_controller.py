"""
Simple posture controller for the Unitree H1 scene.

This is not full locomotion or manipulation control. It holds the H1 at the
home keyframe with gravity-compensated joint torques and swaps between a few
phase-dependent upper-body poses so the rollout reads as an active robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointRef:
    actuator_index: int
    joint_id: int
    qpos_addr: int
    dof_addr: int
    name: str


class H1TaskController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, lock_base: bool = True) -> None:
        self.model = model
        self.data = data
        self.lock_base = lock_base
        self.home_qpos = self._load_home_pose()
        self.base_qpos = self.home_qpos[:7].copy()
        self.target_qpos = self.home_qpos.copy()
        self.phase = "stand"
        self.joints = self._build_joint_refs()
        self.kp = 120.0
        self.kd = 12.0
        self.ctrl_low = np.array(self.model.actuator_ctrlrange[:, 0], dtype=float)
        self.ctrl_high = np.array(self.model.actuator_ctrlrange[:, 1], dtype=float)
        self._lock_base_pose()

    def _load_home_pose(self) -> np.ndarray:
        if self.model.nkey > 0:
            return np.array(self.model.key_qpos[0], dtype=float)
        return np.array(self.data.qpos, dtype=float)

    def _build_joint_refs(self) -> Dict[str, JointRef]:
        refs: Dict[str, JointRef] = {}
        for actuator_index in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_index, 0])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None:
                continue
            refs[joint_name] = JointRef(
                actuator_index=actuator_index,
                joint_id=joint_id,
                qpos_addr=int(self.model.jnt_qposadr[joint_id]),
                dof_addr=int(self.model.jnt_dofadr[joint_id]),
                name=joint_name,
            )
        return refs

    def reset(self) -> None:
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.home_qpos = self._load_home_pose()
        self.base_qpos = self.home_qpos[:7].copy()
        self.target_qpos = self.home_qpos.copy()
        self.phase = "stand"
        self._lock_base_pose()
        mujoco.mj_forward(self.model, self.data)

    def _lock_base_pose(self) -> None:
        if not self.lock_base:
            return
        self.data.qpos[:7] = self.base_qpos
        self.data.qvel[:6] = 0.0

    def _side_from_name(self, name: Optional[str]) -> str:
        if not name:
            return "both"
        if name in {"box_1", "box_2", "back_left", "front_left"}:
            return "left"
        if name in {"box_4", "box_5", "back_right", "front_right"}:
            return "right"
        return "both"

    def _set_joint(self, target: np.ndarray, joint_name: str, value: float) -> None:
        ref = self.joints.get(joint_name)
        if ref is not None:
            target[ref.qpos_addr] = value

    def _apply_arm_pose(self, target: np.ndarray, side: str, *, shoulder_pitch: float, shoulder_roll: float, shoulder_yaw: float, elbow: float) -> None:
        if side in {"left", "both"}:
            self._set_joint(target, "left_shoulder_pitch", shoulder_pitch)
            self._set_joint(target, "left_shoulder_roll", shoulder_roll)
            self._set_joint(target, "left_shoulder_yaw", shoulder_yaw)
            self._set_joint(target, "left_elbow", elbow)
        if side in {"right", "both"}:
            self._set_joint(target, "right_shoulder_pitch", shoulder_pitch)
            self._set_joint(target, "right_shoulder_roll", -shoulder_roll)
            self._set_joint(target, "right_shoulder_yaw", -shoulder_yaw)
            self._set_joint(target, "right_elbow", elbow)

    def set_mode(self, phase: str, box_id: Optional[str] = None, slot_id: Optional[str] = None) -> None:
        target = self.home_qpos.copy()
        self._set_joint(target, "torso", 0.12)

        if phase == "pick":
            side = self._side_from_name(box_id)
            self._apply_arm_pose(target, side, shoulder_pitch=0.72, shoulder_roll=0.22, shoulder_yaw=0.08, elbow=0.88)
        elif phase == "move":
            self._apply_arm_pose(target, "both", shoulder_pitch=0.42, shoulder_roll=0.10, shoulder_yaw=0.00, elbow=0.52)
        elif phase == "place":
            side = self._side_from_name(slot_id)
            self._apply_arm_pose(target, side, shoulder_pitch=0.78, shoulder_roll=0.26, shoulder_yaw=0.06, elbow=0.96)
        else:
            self._apply_arm_pose(target, "both", shoulder_pitch=0.12, shoulder_roll=0.02, shoulder_yaw=0.00, elbow=0.20)

        self.target_qpos = target
        self.phase = phase

    def apply(self) -> None:
        self._lock_base_pose()
        mujoco.mj_forward(self.model, self.data)

        ctrl = np.zeros(self.model.nu, dtype=float)
        for ref in self.joints.values():
            desired = self.target_qpos[ref.qpos_addr]
            current = self.data.qpos[ref.qpos_addr]
            velocity = self.data.qvel[ref.dof_addr]
            bias = self.data.qfrc_bias[ref.dof_addr]
            command = bias + self.kp * (desired - current) - self.kd * velocity
            ctrl[ref.actuator_index] = float(np.clip(command, self.ctrl_low[ref.actuator_index], self.ctrl_high[ref.actuator_index]))

        self.data.ctrl[:] = ctrl

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.apply()
            mujoco.mj_step(self.model, self.data)
            self._lock_base_pose()
            mujoco.mj_forward(self.model, self.data)