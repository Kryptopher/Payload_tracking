#!/usr/bin/env python3
"""
Measure AprilTag layout residuals for the 100 mm cube target.

Hold the cube steady in view for a few seconds. The script reports how
consistent each tag's detector-derived cube center is versus the median
across visible tags. Large per-tag residuals usually mean tag_size_m or
center_m in the layout JSON does not match the physical cube.

Usage:
    python3 calibrate_cube_layout.py --ip 192.168.0.153 \\
        --layout tag_layout_box_100mm.json --duration 5
    python3 calibrate_cube_layout.py --write-output tag_layout_calibrated.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from pupil_apriltags import Detector

from wall_tracker import (
    CUBE_FACE_TAG_GROUPS,
    TagLayout,
    _SuppressStderr,
    _payload_center_camera_from_detection,
)


def _camera_socket(name: str):
    return getattr(dai.CameraBoardSocket, name)


def _ensure_undistort_maps(gray, camera_matrix, dist_coeffs, cache: dict):
    key = (gray.shape, camera_matrix.tobytes(), dist_coeffs.tobytes())
    if cache.get("key") == key:
        return
    h, w = gray.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        camera_matrix,
        (w, h),
        cv2.CV_16SC2,
    )
    cache["key"] = key
    cache["map1"] = map1
    cache["map2"] = map2


def _undistort(gray, cache: dict) -> np.ndarray:
    if "map1" not in cache:
        return gray
    return cv2.remap(gray, cache["map1"], cache["map2"], cv2.INTER_LINEAR)


def run_calibration(args: argparse.Namespace) -> int:
    layout = TagLayout.load(args.layout, tag_size_override=args.tag_size)
    detector = Detector(
        families=layout.tag_family,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        refine_edges=1,
    )
    undist_cache: dict = {}
    per_tag_centers: dict[int, list[np.ndarray]] = defaultdict(list)
    frame_spreads_mm: list[float] = []

    if args.ip:
        os.environ["DEPTHAI_DEVICE_NAME"] = args.ip

    socket_id = _camera_socket(args.camera_socket)
    t_end = time.monotonic() + float(args.duration)
    frames = 0

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(socket_id)
        detect_out = cam.requestOutput(
            (args.detect_width, args.detect_height),
            dai.ImgFrame.Type.GRAY8,
            fps=args.detect_fps,
        )
        detect_q = detect_out.createOutputQueue()
        detect_q.setMaxSize(1)
        detect_q.setBlocking(False)
        pipeline.start()
        device = pipeline.getDefaultDevice()
        calib = device.readCalibration()
        camera_matrix = np.array(
            calib.getCameraIntrinsics(
                socket_id, args.detect_width, args.detect_height
            ),
            dtype=np.float64,
        )
        dist = np.array(
            calib.getDistortionCoefficients(socket_id), dtype=np.float64
        ).reshape(-1, 1)

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])

        print(f"[CAL] Collecting for {args.duration:.1f}s — hold cube steady...")
        while pipeline.isRunning() and time.monotonic() < t_end:
            msg = detect_q.tryGet()
            if msg is None:
                time.sleep(0.002)
                continue
            gray = msg.getCvFrame()
            _ensure_undistort_maps(gray, camera_matrix, dist, undist_cache)
            undist = _undistort(gray, undist_cache)
            with _SuppressStderr():
                raw = detector.detect(
                    undist,
                    estimate_tag_pose=True,
                    camera_params=(fx, fy, cx, cy),
                    tag_size=layout.tag_size_m,
                )
            frame_centers = []
            for det in raw:
                tid = int(det.tag_id)
                if tid not in layout.valid_ids:
                    continue
                if float(det.decision_margin) < args.decision_margin_min:
                    continue
                tag = layout.tags[tid]
                center = _payload_center_camera_from_detection(
                    det, tag.center_m, tag.orientation_m
                )
                if center is None or not np.all(np.isfinite(center)):
                    continue
                per_tag_centers[tid].append(center)
                frame_centers.append(center)
            if len(frame_centers) >= 2:
                med = np.median(np.vstack(frame_centers), axis=0)
                spread = np.linalg.norm(
                    np.vstack(frame_centers) - med, axis=1
                )
                frame_spreads_mm.append(float(np.median(spread) * 1000.0))
            frames += 1

    if frames == 0:
        print("[CAL] No frames captured.")
        return 1

    print(f"\n[CAL] Frames: {frames}")
    if frame_spreads_mm:
        print(
            f"[CAL] Median cross-tag spread per frame: "
            f"{np.median(frame_spreads_mm):.2f} mm "
            f"(p95 {np.percentile(frame_spreads_mm, 95):.2f} mm)"
        )

    global_samples = []
    for samples in per_tag_centers.values():
        global_samples.extend(samples)
    if not global_samples:
        print("[CAL] No valid tag detections.")
        return 1
    global_median = np.median(np.vstack(global_samples), axis=0)

    print("\n[CAL] Per-tag residual vs global median (mm):")
    residuals_mm: dict[int, float] = {}
    for tid in sorted(per_tag_centers):
        pts = np.vstack(per_tag_centers[tid])
        tag_median = np.median(pts, axis=0)
        residual = float(np.linalg.norm(tag_median - global_median) * 1000.0)
        residuals_mm[tid] = residual
        n = len(per_tag_centers[tid])
        print(f"  ID{tid:2d}: n={n:4d}  residual={residual:6.2f} mm")

    worst = max(residuals_mm.values()) if residuals_mm else 0.0
    print(f"\n[CAL] Worst tag residual: {worst:.2f} mm")
    if worst <= args.residual_ok_mm:
        print(f"[CAL] Layout looks consistent (<= {args.residual_ok_mm:.1f} mm).")
    else:
        print(
            f"[CAL] Layout may need adjustment (>{args.residual_ok_mm:.1f} mm). "
            "Check tag_size_m and center_m."
        )

    if args.write_output is not None:
        out_path = Path(args.write_output)
        with args.layout.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data = copy.deepcopy(data)
        data["name"] = str(data.get("name", "cube")) + "_calibrated"
        data["calibration_note"] = (
            f"Residuals mm: {residuals_mm}; worst={worst:.2f}; "
            f"frames={frames}"
        )
        if worst > args.residual_ok_mm:
            data["notes"] = list(data.get("notes", [])) + [
                "Calibration detected large residuals — verify physical tag size "
                f"({layout.tag_size_m * 1000:.0f} mm) and face positions."
            ]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"[CAL] Wrote {out_path}")

    return 0 if worst <= args.residual_ok_mm else 2


def build_arg_parser() -> argparse.ArgumentParser:
    default_layout = Path(__file__).with_name("tag_layout_box_100mm.json")
    parser = argparse.ArgumentParser(description="Calibrate cube AprilTag layout")
    parser.add_argument("--ip", default=os.environ.get("OAK_IP", "192.168.0.153"))
    parser.add_argument("--camera-socket", default="CAM_B")
    parser.add_argument("--layout", type=Path, default=default_layout)
    parser.add_argument("--tag-size", type=float, default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--detect-width", type=int, default=640)
    parser.add_argument("--detect-height", type=int, default=400)
    parser.add_argument("--detect-fps", type=int, default=30)
    parser.add_argument("--quad-decimate", type=float, default=1.5)
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--decision-margin-min", type=float, default=8.0)
    parser.add_argument(
        "--residual-ok-mm",
        type=float,
        default=3.0,
        help="Residual threshold for pass / layout OK message",
    )
    parser.add_argument(
        "--write-output",
        type=Path,
        default=None,
        help="Optional path to write annotated layout JSON",
    )
    return parser


def main() -> int:
    return run_calibration(build_arg_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
