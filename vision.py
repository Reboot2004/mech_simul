"""
Day 2: top-down vision pipeline for the truck loading simulator.

This module bridges the rendered MuJoCo top-down frame to a YOLOv8-nano
interface. If Ultralytics or YOLO weights are not available, it falls back to a
simple color-based detector so the integration remains runnable in a minimal
workspace.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

from env import BOX_META, TruckLoadingEnv


@dataclass
class Detection:
    box_id: str
    label: str
    confidence: float
    bbox: List[float]
    source: str


COLOR_RANGES: Dict[str, Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "red": [((0, 70, 60), (12, 255, 255)), ((168, 70, 60), (180, 255, 255))],
    "orange": [((10, 90, 80), (30, 255, 255))],
    "green": [((35, 60, 60), (90, 255, 255))],
}


class TopDownVisionPipeline:
    def __init__(
        self,
        env: TruckLoadingEnv,
        weights_path: Optional[str] = None,
        confidence: float = 0.25,
        iou: float = 0.45,
        auto_download_weights: bool = False,
    ) -> None:
        self.env = env
        self.weights_path = weights_path or "yolov8n.pt"
        self.confidence = confidence
        self.iou = iou
        self.auto_download_weights = auto_download_weights
        self.model = self._load_model(self.weights_path)
        self.last_source = "color_fallback"
        self.fallback_used = True

    def _load_model(self, weights_path: Optional[str]):
        if YOLO is None or not weights_path:
            return None

        try:
            return YOLO(str(weights_path))
        except Exception:
            return None

    def capture_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        return self.env.render_topdown(width=width, height=height)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is not None:
            detections = self._detect_with_yolo(frame)
            if detections:
                self.last_source = "yolo"
                self.fallback_used = False
                return detections

        detections = self._detect_by_color(frame)
        self.last_source = "color_fallback" if self.model is None else "yolo+color_fallback"
        self.fallback_used = True
        return detections

    def _detect_with_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, conf=self.confidence, iou=self.iou, verbose=False)
        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", {}) or {}
        detections: List[Detection] = []

        if result.boxes is None:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

        for index, bbox in enumerate(boxes_xyxy):
            cls_id = int(class_ids[index]) if len(class_ids) > index else -1
            label = names.get(cls_id, f"class_{cls_id}")
            box_id = label if label in BOX_META else label
            confidence = float(confidences[index]) if len(confidences) > index else 0.0
            detections.append(
                Detection(
                    box_id=box_id,
                    label=label,
                    confidence=confidence,
                    bbox=[float(value) for value in bbox.tolist()],
                    source="yolo",
                )
            )

        return detections

    def _detect_by_color(self, frame: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        raw_components: List[Dict[str, Any]] = []

        for color_name, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 300:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                raw_components.append(
                    {
                        "color": color_name,
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "center": (x + w / 2.0, y + h / 2.0),
                        "area": float(area),
                    }
                )

        raw_components.sort(key=lambda item: (item["center"][0], item["center"][1]))
        ordered_box_ids = list(BOX_META.keys())
        detections: List[Detection] = []

        for index, component in enumerate(raw_components):
            if index < len(ordered_box_ids):
                box_id = ordered_box_ids[index]
            else:
                box_id = f"box_{index + 1}"

            detections.append(
                Detection(
                    box_id=box_id,
                    label=component["color"],
                    confidence=0.85,
                    bbox=component["bbox"],
                    source="color",
                )
            )

        return detections

    def annotate(self, frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        palette = {
            "yolo": (0, 220, 0),
            "color": (0, 165, 255),
        }

        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
            color = palette.get(detection.source, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            caption = f"{detection.box_id} {detection.label} {detection.confidence:.2f}"
            text_origin = (x1, max(18, y1 - 6))
            cv2.putText(
                annotated,
                caption,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def run_once(self, output_dir: str = "outputs/day2") -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = self.capture_frame()
        detections = self.detect(frame)
        annotated = self.annotate(frame, detections)

        frame_file = output_path / "topdown_frame.png"
        annotated_file = output_path / "topdown_annotated.png"
        json_file = output_path / "detections.json"

        cv2.imwrite(str(frame_file), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(annotated_file), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        payload = {
            "source": self.last_source,
            "fallback_used": self.fallback_used,
            "model_name": self.weights_path,
            "weights_path": self.weights_path,
            "detections": [asdict(detection) for detection in detections],
        }
        json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return {
            "frame_file": str(frame_file),
            "annotated_file": str(annotated_file),
            "json_file": str(json_file),
            "detections": payload["detections"],
            "source": payload["source"],
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 2 top-down vision pipeline")
    parser.add_argument("--xml", default="scene.xml", help="Path to the MuJoCo scene XML")
    parser.add_argument("--weights", default=None, help="Path to YOLOv8-nano weights")
    parser.add_argument("--output-dir", default="outputs/day2", help="Directory for outputs")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument(
        "--auto-download-weights",
        action="store_true",
        help="Allow Ultralytics to fetch default weights if they are not present locally",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    env = TruckLoadingEnv(xml_path=args.xml)
    pipeline = TopDownVisionPipeline(
        env=env,
        weights_path=args.weights,
        confidence=args.confidence,
        iou=args.iou,
        auto_download_weights=args.auto_download_weights,
    )
    report = pipeline.run_once(output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
