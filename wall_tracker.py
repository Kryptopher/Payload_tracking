#!/usr/bin/env python3
"""
Standalone OAK-D S2 PoE AprilTag wall/target tracker.

This program is intentionally independent of the ROS payload tracker. It uses a
fixed OAK camera, a rigid AprilTag layout on the moving target, and one PnP
solve per detection frame to estimate translation and rotation together.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import cv2
import depthai as dai
import numpy as np
from pupil_apriltags import Detector


class _SuppressStderr:
    def __enter__(self):
        sys.stderr.flush()
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._saved = os.dup(2)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self._saved, 2)
        os.close(self._devnull)
        os.close(self._saved)


class MJPEGStream:
    def __init__(self):
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()

    def update(self, frame: np.ndarray, quality: int = 75) -> None:
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        )
        if not ok:
            return
        with self._lock:
            self._frame = jpeg.tobytes()

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._frame


stream = MJPEGStream()


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""<html><head>
                <title>Wall Tracker</title>
                <style>
                    body { background:#111; color:#ddd; margin:0;
                           font-family:sans-serif; }
                    .wrap { display:flex; justify-content:center;
                            align-items:center; height:100vh; }
                    img { max-width:100%; max-height:100vh; }
                </style>
            </head><body>
                <div class="wrap"><img src="/stream"/></div>
            </body></html>"""
            )
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            while True:
                frame = stream.get()
                if frame is None:
                    time.sleep(0.01)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        super().server_bind()


def start_stream_server(port: int) -> Optional[HTTPServer]:
    try:
        server = ReusableHTTPServer(("0.0.0.0", int(port)), StreamHandler)
    except OSError as exc:
        if exc.errno in (48, 98):
            print(f"[STREAM] Port {port} is busy; running without MJPEG.")
            return None
        raise
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def local_lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


@dataclass(frozen=True)
class TagSpec:
    tag_id: int
    center_m: np.ndarray
    orientation_m: np.ndarray
    rotation_deg: float = 0.0


def _rotation_matrix_from_rpy_deg(rpy_deg: list[float] | tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = [math.radians(float(v)) for v in rpy_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _tag_orientation_from_normal_up(normal: list[float], up: list[float]) -> np.ndarray:
    normal_v = np.asarray(normal, dtype=np.float64).reshape(3)
    up_v = np.asarray(up, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(normal_v))
    u_norm = float(np.linalg.norm(up_v))
    if n_norm <= 1e-9 or u_norm <= 1e-9:
        raise ValueError("tag normal_m and up_m must be non-zero vectors")
    normal_v = normal_v / n_norm
    up_v = up_v / u_norm
    right_v = np.cross(up_v, normal_v)
    r_norm = float(np.linalg.norm(right_v))
    if r_norm <= 1e-9:
        raise ValueError("tag normal_m and up_m must not be parallel")
    right_v = right_v / r_norm
    up_v = np.cross(normal_v, right_v)
    up_v = up_v / max(1e-9, float(np.linalg.norm(up_v)))
    return np.column_stack([right_v, up_v, normal_v]).astype(np.float64)


class TagLayout:
    def __init__(
        self,
        *,
        tag_size_m: float,
        tag_family: str,
        tags: list[TagSpec],
        target_center_m: Optional[np.ndarray] = None,
        name: str = "wall_target",
    ):
        if tag_size_m <= 0:
            raise ValueError("tag_size_m must be positive")
        if len(tags) < 3:
            raise ValueError("layout should contain at least three tags")
        self.name = name
        self.tag_size_m = float(tag_size_m)
        self.tag_family = tag_family
        self.tags = {tag.tag_id: tag for tag in tags}
        self.target_center_m = (
            np.asarray(target_center_m, dtype=np.float64).reshape(3)
            if target_center_m is not None
            else np.zeros(3, dtype=np.float64)
        )

    @property
    def valid_ids(self) -> set[int]:
        return set(self.tags.keys())

    @classmethod
    def load(cls, path: Path, tag_size_override: Optional[float] = None) -> "TagLayout":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        tag_size = float(
            tag_size_override
            if tag_size_override is not None
            else data.get("tag_size_m", data.get("marker_size_m", 0.10))
        )
        family = str(data.get("tag_family", "tagStandard41h12"))
        tags = []
        for item in data["tags"]:
            center = item["center_m"]
            if len(center) == 2:
                center = [center[0], center[1], 0.0]
            if len(center) != 3:
                raise ValueError(f"tag {item.get('id')} center_m must have 2 or 3 values")
            rotation_deg = float(item.get("rotation_deg", 0.0))
            if "normal_m" in item and "up_m" in item:
                orientation = _tag_orientation_from_normal_up(item["normal_m"], item["up_m"])
            elif "rotation_rpy_deg" in item:
                orientation = _rotation_matrix_from_rpy_deg(item["rotation_rpy_deg"])
            else:
                orientation = _rotation_matrix_from_rpy_deg([0.0, 0.0, rotation_deg])
            tags.append(
                TagSpec(
                    tag_id=int(item["id"]),
                    center_m=np.array(center, dtype=np.float64),
                    orientation_m=orientation,
                    rotation_deg=rotation_deg,
                )
            )
        target_center = data.get("target_center_m", [0.0, 0.0, 0.0])
        if len(target_center) == 2:
            target_center = [target_center[0], target_center[1], 0.0]
        return cls(
            tag_size_m=tag_size,
            tag_family=family,
            tags=tags,
            target_center_m=np.array(target_center, dtype=np.float64),
            name=str(data.get("name", "wall_target")),
        )

    @property
    def is_planar(self) -> bool:
        points = [self.object_corners(tag_id) for tag_id in self.tags]
        obj = np.vstack(points).reshape(-1, 3)
        return _points_are_coplanar(obj)

    def object_corners(self, tag_id: int) -> np.ndarray:
        tag = self.tags[tag_id]
        h = 0.5 * self.tag_size_m
        local = np.array(
            [
                [-h, h, 0.0],
                [h, h, 0.0],
                [h, -h, 0.0],
                [-h, -h, 0.0],
            ],
            dtype=np.float64,
        )
        return tag.center_m.reshape(1, 3) + local @ tag.orientation_m.T


class MedianEmaFilter:
    def __init__(
        self,
        dim: int,
        *,
        median_window: int,
        alpha: float,
        max_step: float,
        deadband: float = 0.0,
    ):
        self.dim = int(dim)
        self.window = max(1, int(median_window))
        self.alpha = float(alpha)
        self.max_step = float(max_step)
        self.deadband = float(deadband)
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)
        self._value: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._buf.clear()
        self._value = None

    def filter(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).reshape(self.dim)
        self._buf.append(v)
        if len(self._buf) >= 3:
            candidate = np.median(np.vstack(self._buf), axis=0)
        else:
            candidate = v

        if self._value is None:
            self._value = candidate.copy()
            return self._value.copy()

        step = candidate - self._value
        max_abs = float(np.max(np.abs(step)))
        if self.max_step > 0 and max_abs > self.max_step:
            candidate = self._value + step * (self.max_step / max_abs)

        if self.deadband > 0.0:
            if float(np.linalg.norm(candidate - self._value)) < self.deadband:
                return self._value.copy()

        self._value = self.alpha * candidate + (1.0 - self.alpha) * self._value
        return self._value.copy()


class SessionReference:
    def __init__(self):
        self.t0: Optional[np.ndarray] = None
        self.r0: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.t0 = None
        self.r0 = None

    def relative(self, tvec: np.ndarray, rmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        r = np.asarray(rmat, dtype=np.float64).reshape(3, 3)
        if self.t0 is None or self.r0 is None:
            self.t0 = t.copy()
            self.r0 = r.copy()
        return t - self.t0, r @ self.r0.T


@dataclass
class PoseEstimate:
    rvec: np.ndarray
    tvec: np.ndarray
    rmat: np.ndarray
    rel_t: np.ndarray
    rel_t_smooth: np.ndarray
    operator_abs: np.ndarray
    operator_abs_smooth: np.ndarray
    operator_t: np.ndarray
    operator_t_smooth: np.ndarray
    rel_euler_deg: np.ndarray
    rel_euler_smooth_deg: np.ndarray
    omega_rad_s: float
    omega_deg_s: float
    depth_m: float
    depth_source: str
    pose_source: str
    center_source: str
    origin_status: str
    target_center_px: tuple[float, float]
    visible_center_px: tuple[float, float]
    reproj_error_px: float
    tag_ids: list[int]
    primary_face_ids: tuple[int, ...]
    center_err_mm: float
    corner_count: int
    status: str


@dataclass
class FrameResult:
    detections: list
    pose: Optional[PoseEstimate]
    status: str
    message: str
    timestamp: float


def rotation_matrix_to_euler_xyz_deg(r: np.ndarray) -> np.ndarray:
    """Return roll(X), pitch(Y), yaw(Z) for R = Rz * Ry * Rx."""
    r = np.asarray(r, dtype=np.float64).reshape(3, 3)
    sy = -float(r[2, 0])
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-6:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    return np.degrees([roll, pitch, yaw])


def rotation_delta_rad(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest 3D rotation angle between two rotation matrices."""
    r = np.asarray(a, dtype=np.float64).reshape(3, 3) @ np.asarray(
        b, dtype=np.float64
    ).reshape(3, 3).T
    c = 0.5 * (float(np.trace(r)) - 1.0)
    return math.acos(float(np.clip(c, -1.0, 1.0)))


def wrapped_angle_delta_deg(new_deg: float, old_deg: float) -> float:
    """Signed smallest delta from old angle to new angle in degrees."""
    return (float(new_deg) - float(old_deg) + 180.0) % 360.0 - 180.0


def pixel_depth_to_operator_xyz(
    u_px: float,
    v_px: float,
    depth_m: float,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Camera pixel plus forward depth -> operator [X lateral, Y forward, Z vertical]."""
    k = np.asarray(camera_matrix, dtype=np.float64)
    z = float(depth_m)
    x = (float(u_px) - float(k[0, 2])) / float(k[0, 0]) * z
    vertical = (float(v_px) - float(k[1, 2])) / float(k[1, 1]) * z
    return np.array([x, z, vertical], dtype=np.float64)


def pnp_tvec_to_operator_xyz(tvec: np.ndarray) -> np.ndarray:
    """OpenCV camera tvec [X, Y, Z-depth] -> operator [X, Y-forward, Z-vertical]."""
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return np.array([float(t[0]), float(t[2]), float(t[1])], dtype=np.float64)


def pixel_depth_to_camera_xyz(
    u_px: float,
    v_px: float,
    depth_m: float,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Camera pixel plus forward depth -> OpenCV camera [X, Y, Z-depth]."""
    op = pixel_depth_to_operator_xyz(u_px, v_px, depth_m, camera_matrix)
    return np.array([op[0], op[2], op[1]], dtype=np.float64)


def _points_are_coplanar(points: np.ndarray, tol_m: float = 1e-5) -> bool:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 4:
        return True
    centered = pts - np.mean(pts, axis=0)
    _u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    if s.size < 3:
        return True
    return float(s[-1]) <= float(tol_m)


def _target_center_camera(
    rmat: np.ndarray,
    tvec: np.ndarray,
    target_center_m: np.ndarray,
) -> np.ndarray:
    r = np.asarray(rmat, dtype=np.float64).reshape(3, 3)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    c = np.asarray(target_center_m, dtype=np.float64).reshape(3)
    return r @ c + t


def _payload_center_camera_from_detection(
    det,
    tag_center_m: np.ndarray,
    tag_orientation_m: np.ndarray,
) -> Optional[np.ndarray]:
    """Match payload_tracker ArucoTracker._payload_position()."""
    if not hasattr(det, "pose_R") or not hasattr(det, "pose_t"):
        return None
    R_pc = np.asarray(det.pose_R, dtype=np.float64) @ np.asarray(
        tag_orientation_m, dtype=np.float64
    ).T
    return np.asarray(det.pose_t, dtype=np.float64).reshape(3) - R_pc @ np.asarray(
        tag_center_m, dtype=np.float64
    ).reshape(3)


# Complete face groups for the 100 mm cube layout (3 tags per vertical face).
CUBE_FACE_TAG_GROUPS = (
    frozenset({0, 1, 2}),
    frozenset({3, 4, 5}),
    frozenset({6, 7, 8}),
    frozenset({9, 10, 11}),
)


class RigidTagPoseEstimator:
    def __init__(
        self,
        layout: TagLayout,
        *,
        decision_margin_min: float,
        quad_decimate: float,
        nthreads: int,
        min_pose_tags: int,
        max_reproj_error_px: float,
        pose_alpha: float,
        angle_alpha: float,
        median_window: int,
        max_translation_step_m: float,
        max_angle_step_deg: float,
        depth_roi_px: int,
        depth_min_m: float,
        depth_max_m: float,
        single_tag_fallback: bool,
        graph_deadband_m: float,
    ):
        self.layout = layout
        self.decision_margin_min = float(decision_margin_min)
        self.min_pose_tags = max(1, int(min_pose_tags))
        self.max_reproj_error_px = float(max_reproj_error_px)
        self.detector = Detector(
            families=layout.tag_family,
            nthreads=int(nthreads),
            quad_decimate=float(quad_decimate),
            refine_edges=1,
        )
        self.session = SessionReference()
        self.translation_filter = MedianEmaFilter(
            3,
            median_window=median_window,
            alpha=pose_alpha,
            max_step=max_translation_step_m,
        )
        self.operator_abs_filter = MedianEmaFilter(
            3,
            median_window=median_window,
            alpha=min(pose_alpha, 0.22),
            max_step=max_translation_step_m,
            deadband=graph_deadband_m,
        )
        self.operator_filter = MedianEmaFilter(
            3,
            median_window=median_window,
            alpha=pose_alpha,
            max_step=max_translation_step_m,
        )
        self.angle_filter = MedianEmaFilter(
            3,
            median_window=median_window,
            alpha=angle_alpha,
            max_step=max_angle_step_deg,
        )
        self.depth_roi_px = max(1, int(depth_roi_px))
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.single_tag_fallback = bool(single_tag_fallback)
        self._operator_origin: Optional[np.ndarray] = None
        self._origin_candidates: deque[np.ndarray] = deque(maxlen=8)
        self._origin_warmup_min = 4
        self._origin_max_spread_m = 0.08
        self._prev_abs_tvec: Optional[np.ndarray] = None
        self._prev_target_center_camera: Optional[np.ndarray] = None
        self._prev_abs_rmat: Optional[np.ndarray] = None
        self._prev_yaw_deg: Optional[float] = None
        self._prev_unwrapped_yaw_deg: Optional[float] = None
        self._prev_omega_ts: Optional[float] = None
        self._omega_deg_s = 0.0
        self._detector_center_cam: Optional[np.ndarray] = None
        self._primary_face_ids: Optional[frozenset[int]] = None
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._intrinsics_key: Optional[tuple] = None

    def reset_session(self) -> None:
        self.session.reset()
        self.translation_filter.reset()
        self.operator_abs_filter.reset()
        self.operator_filter.reset()
        self.angle_filter.reset()
        self._operator_origin = None
        self._origin_candidates.clear()
        self._prev_abs_tvec = None
        self._prev_target_center_camera = None
        self._prev_abs_rmat = None
        self._prev_yaw_deg = None
        self._prev_unwrapped_yaw_deg = None
        self._prev_omega_ts = None
        self._omega_deg_s = 0.0
        self._detector_center_cam = None
        self._primary_face_ids = None

    def _ensure_undistort_maps(
        self,
        gray: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        key = (gray.shape, camera_matrix.tobytes(), dist_coeffs.tobytes())
        if self._intrinsics_key == key:
            return
        h, w = gray.shape[:2]
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )
        self._intrinsics_key = key

    def _undistort_gray(self, gray: np.ndarray) -> np.ndarray:
        if self._map1 is None or self._map2 is None:
            return gray
        return cv2.remap(gray, self._map1, self._map2, cv2.INTER_LINEAR)

    def _detector_center_from_detections(
        self,
        detections: list,
        face_ids: Optional[frozenset[int]] = None,
    ) -> Optional[np.ndarray]:
        estimates = []
        for det in detections:
            if face_ids is not None and int(det.tag_id) not in face_ids:
                continue
            tag = self.layout.tags.get(int(det.tag_id))
            if tag is None:
                continue
            center = _payload_center_camera_from_detection(
                det, tag.center_m, tag.orientation_m
            )
            if center is not None and np.all(np.isfinite(center)):
                estimates.append(center)
        if not estimates:
            return None
        return np.median(np.vstack(estimates), axis=0)

    def _select_primary_face(
        self,
        detections: list,
        tag_ids: list[int],
    ) -> Optional[frozenset[int]]:
        detected = frozenset(int(t) for t in tag_ids)
        by_id = {int(det.tag_id): det for det in detections}
        candidates: list[tuple[float, frozenset[int]]] = []
        for face_ids in CUBE_FACE_TAG_GROUPS:
            visible = face_ids & detected
            if len(visible) < 2:
                continue
            margin_sum = sum(float(by_id[tid].decision_margin) for tid in visible)
            score = 100.0 * float(len(visible)) + margin_sum
            if self._primary_face_ids is not None and face_ids == self._primary_face_ids:
                score += 50.0
            candidates.append((score, face_ids))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        self._primary_face_ids = candidates[0][1]
        return self._primary_face_ids

    @staticmethod
    def _rvec_from_detection(det, tag_orientation_m: np.ndarray) -> np.ndarray:
        R_cb = np.asarray(det.pose_R, dtype=np.float64) @ np.asarray(
            tag_orientation_m, dtype=np.float64
        ).T
        rvec, _ = cv2.Rodrigues(R_cb)
        return rvec

    def _best_detection(
        self,
        detections: list,
        prefer_ids: Optional[frozenset[int]] = None,
    ):
        pool = detections
        if prefer_ids:
            filtered = [d for d in detections if int(d.tag_id) in prefer_ids]
            if filtered:
                pool = filtered
        return max(pool, key=lambda d: float(d.decision_margin))

    def _estimate_pose(
        self,
        detections: list,
        tag_ids: list[int],
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
    ) -> tuple[Optional[tuple[np.ndarray, np.ndarray, float, str]], tuple[int, ...]]:
        detected = frozenset(int(t) for t in tag_ids)
        primary_face = self._select_primary_face(detections, tag_ids)
        primary_visible = tuple(
            sorted((primary_face & detected) if primary_face else detected)
        )
        face_filter = primary_face if primary_face else None
        self._detector_center_cam = self._detector_center_from_detections(
            detections, face_filter
        )

        solved: Optional[tuple[np.ndarray, np.ndarray, float, str]] = None

        if primary_face is not None and len(primary_face & detected) == 3:
            face_dets = [d for d in detections if int(d.tag_id) in primary_face]
            solved = self._solve_face_pose(
                face_dets, camera_matrix, dist_coeffs, "face multi-tag"
            )
            if solved is not None and (
                not math.isfinite(solved[2]) or solved[2] > self.max_reproj_error_px
            ):
                solved = None

        if solved is None and len(primary_visible) >= 2:
            face_dets = [
                d for d in detections if int(d.tag_id) in primary_visible
            ]
            pair_dets = sorted(
                face_dets, key=lambda d: -float(d.decision_margin)
            )[:2]
            solved = self._solve_face_pose(
                pair_dets, camera_matrix, dist_coeffs, "face pair"
            )
            if solved is not None and (
                not math.isfinite(solved[2]) or solved[2] > self.max_reproj_error_px
            ):
                solved = None

        if solved is None and self._detector_center_cam is not None:
            prefer = frozenset(primary_visible) if primary_visible else None
            best = self._best_detection(detections, prefer)
            tag = self.layout.tags[int(best.tag_id)]
            rvec = self._rvec_from_detection(best, tag.orientation_m)
            obj = self.layout.object_corners(int(best.tag_id)).reshape(-1, 3)
            img = np.asarray(best.corners, dtype=np.float64).reshape(-1, 2)
            reproj = self._rms_reprojection(
                obj,
                img,
                rvec,
                self._detector_center_cam.reshape(3, 1),
                camera_matrix,
                dist_coeffs,
            )
            solved = (
                rvec,
                self._detector_center_cam.reshape(3, 1),
                reproj,
                "detector-center",
            )

        if solved is None:
            pool = [
                d for d in detections if int(d.tag_id) in primary_visible
            ] if primary_visible else detections
            solved = self._solve_single_tag_pose(pool, camera_matrix, dist_coeffs)
            if solved is None and pool is not detections:
                solved = self._solve_single_tag_pose(
                    detections, camera_matrix, dist_coeffs
                )

        return solved, primary_visible

    def process(
        self,
        gray: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        timestamp: float,
        depth_frame: Optional[np.ndarray] = None,
    ) -> FrameResult:
        self._ensure_undistort_maps(gray, camera_matrix, dist_coeffs)
        undist = self._undistort_gray(gray)
        pnp_dist: Optional[np.ndarray] = None

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])

        with _SuppressStderr():
            raw = self.detector.detect(
                undist,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=self.layout.tag_size_m,
            )

        detections = []
        for det in raw:
            if int(det.tag_id) not in self.layout.valid_ids:
                continue
            if float(det.decision_margin) < self.decision_margin_min:
                continue
            detections.append(det)

        tag_ids = sorted({int(det.tag_id) for det in detections})
        if not tag_ids:
            return FrameResult(
                detections=detections,
                pose=None,
                status="REACQ",
                message=f"{len(tag_ids)} tag(s); need 1",
                timestamp=timestamp,
            )

        solved, primary_visible = self._estimate_pose(
            detections, tag_ids, camera_matrix, pnp_dist
        )
        if solved is None:
            return FrameResult(
                detections=detections,
                pose=None,
                status="BAD",
                message="PnP failed",
                timestamp=timestamp,
            )

        rvec, tvec, reproj, pose_source = solved
        if not math.isfinite(reproj):
            reproj = float("inf")

        pnp_center_before = None
        center_err_mm = 0.0
        if self._detector_center_cam is not None:
            rmat_check, _ = cv2.Rodrigues(rvec)
            pnp_center_before = _target_center_camera(
                rmat_check, tvec, self.layout.target_center_m
            )
            center_err_mm = float(
                np.linalg.norm(pnp_center_before - self._detector_center_cam) * 1000.0
            )
            tvec = self._detector_center_cam.reshape(3, 1)

        if pose_source.startswith("single"):
            prefer = frozenset(primary_visible) if primary_visible else None
            best = self._best_detection(detections, prefer)
            tag = self.layout.tags[int(best.tag_id)]
            if hasattr(best, "pose_R") and best.pose_R is not None:
                rvec = self._rvec_from_detection(best, tag.orientation_m)

        pose_dets = [
            d for d in detections if int(d.tag_id) in primary_visible
        ] if primary_visible else detections
        object_points, image_points = self._collect_points(pose_dets)

        rmat, _ = cv2.Rodrigues(rvec)
        rel_t, rel_r = self.session.relative(tvec, rmat)
        rel_euler = rotation_matrix_to_euler_xyz_deg(rel_r)
        rel_t_smooth = self.translation_filter.filter(rel_t)
        visible_center_px = self._visible_center_px(detections)
        target_center_px, center_source = self._display_center_px(
            rvec, tvec, camera_matrix, pnp_dist, visible_center_px, pose_source
        )
        visible_object_center_m = np.mean(object_points.reshape(-1, 3), axis=0)
        operator_abs, depth_m, depth_source = self._operator_translation(
            tvec,
            rmat,
            target_center_px,
            visible_center_px,
            visible_object_center_m,
            camera_matrix,
            depth_frame,
        )
        operator_abs_smooth = self.operator_abs_filter.filter(operator_abs)
        operator_t, origin_status = self._operator_relative(
            operator_abs,
            pose_source=pose_source,
            depth_source=depth_source,
        )
        operator_t_smooth = self.operator_filter.filter(operator_t)
        rel_euler_for_filter = rel_euler.copy()
        if self._prev_yaw_deg is not None and self._prev_unwrapped_yaw_deg is not None:
            rel_euler_for_filter[2] = (
                self._prev_unwrapped_yaw_deg
                + wrapped_angle_delta_deg(rel_euler[2], self._prev_yaw_deg)
            )
        rel_euler_smooth = self.angle_filter.filter(rel_euler_for_filter)
        omega_deg_s, omega_rad_s = self._update_omega(
            float(rel_euler_smooth[2]), timestamp
        )
        self._prev_abs_tvec = np.asarray(tvec, dtype=np.float64).reshape(3).copy()
        self._prev_target_center_camera = _target_center_camera(
            rmat, tvec, self.layout.target_center_m
        )
        self._prev_abs_rmat = rmat.copy()
        status = "GOOD"
        if pose_source.startswith("single"):
            status = "WARN"
        elif center_err_mm > 5.0:
            status = "WARN"
        elif pose_source == "face pair":
            status = "WARN"
        pose = PoseEstimate(
            rvec=rvec.reshape(3, 1),
            tvec=tvec.reshape(3, 1),
            rmat=rmat,
            rel_t=rel_t,
            rel_t_smooth=rel_t_smooth,
            operator_abs=operator_abs,
            operator_abs_smooth=operator_abs_smooth,
            operator_t=operator_t,
            operator_t_smooth=operator_t_smooth,
            rel_euler_deg=rel_euler,
            rel_euler_smooth_deg=rel_euler_smooth,
            omega_rad_s=omega_rad_s,
            omega_deg_s=omega_deg_s,
            depth_m=depth_m,
            depth_source=depth_source,
            pose_source=pose_source,
            center_source=center_source,
            origin_status=origin_status,
            target_center_px=target_center_px,
            visible_center_px=visible_center_px,
            reproj_error_px=reproj,
            tag_ids=tag_ids,
            primary_face_ids=tuple(primary_visible),
            center_err_mm=center_err_mm,
            corner_count=int(len(image_points)),
            status=status,
        )
        face_s = (
            f" face={list(primary_visible)}"
            if primary_visible
            else ""
        )
        return FrameResult(
            detections=detections,
            pose=pose,
            status=status,
            message=(
                f"{len(tag_ids)} tag(s), {reproj:.1f}px, "
                f"center_err={center_err_mm:.1f}mm, {pose_source}{face_s}"
            ),
            timestamp=timestamp,
        )

    def _collect_points(self, detections: list) -> tuple[np.ndarray, np.ndarray]:
        object_points = []
        image_points = []
        for det in detections:
            object_points.append(self.layout.object_corners(int(det.tag_id)))
            image_points.append(np.asarray(det.corners, dtype=np.float64))
        obj = np.vstack(object_points).reshape(-1, 3)
        img = np.vstack(image_points).reshape(-1, 2)
        return obj, img

    def _solve_face_pose(
        self,
        face_detections: list,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
        source_label: str,
    ) -> Optional[tuple[np.ndarray, np.ndarray, float, str]]:
        if len(face_detections) < 2:
            return None
        object_points, image_points = self._collect_points(face_detections)
        solved = self._solve_pose(
            object_points, image_points, camera_matrix, dist_coeffs
        )
        if solved is None:
            return None
        rvec, tvec, reproj, _source = solved
        return rvec, tvec, reproj, source_label

    @staticmethod
    def _visible_center_px(detections: list) -> tuple[float, float]:
        corners = [np.asarray(det.corners, dtype=np.float64).reshape(-1, 2) for det in detections]
        pts = np.vstack(corners)
        c = np.mean(pts, axis=0)
        return float(c[0]), float(c[1])

    def _display_center_px(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
        visible_center_px: tuple[float, float],
        pose_source: str,
    ) -> tuple[tuple[float, float], str]:
        if not pose_source.startswith("single") or not self.layout.is_planar:
            pts, _ = cv2.projectPoints(
                self.layout.target_center_m.reshape(1, 3),
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            p = pts.reshape(-1, 2)[0]
            return (float(p[0]), float(p[1])), "layout"
        return visible_center_px, "visible-tags"

    def _operator_translation(
        self,
        tvec: np.ndarray,
        rmat: np.ndarray,
        target_center_px: tuple[float, float],
        visible_center_px: tuple[float, float],
        visible_object_center_m: np.ndarray,
        camera_matrix: np.ndarray,
        depth_frame: Optional[np.ndarray],
    ) -> tuple[np.ndarray, float, str]:
        depth_m = self._sample_depth_m(depth_frame, visible_center_px)
        if depth_m is not None:
            target_cam = _target_center_camera(rmat, tvec, self.layout.target_center_m)
            visible_cam = _target_center_camera(rmat, tvec, visible_object_center_m)
            center_depth_m = float(depth_m + (target_cam[2] - visible_cam[2]))
            if not math.isfinite(center_depth_m) or center_depth_m <= 0.0:
                center_depth_m = float(depth_m)
            return (
                pixel_depth_to_operator_xyz(
                    target_center_px[0], target_center_px[1], center_depth_m, camera_matrix
                ),
                center_depth_m,
                "stereo",
            )

        target_cam = _target_center_camera(rmat, tvec, self.layout.target_center_m)
        fallback = pnp_tvec_to_operator_xyz(target_cam)
        return fallback, float(fallback[1]), "pnp fallback"

    def _operator_relative(
        self,
        operator_abs: np.ndarray,
        *,
        pose_source: str,
        depth_source: str,
    ) -> tuple[np.ndarray, str]:
        p = np.asarray(operator_abs, dtype=np.float64).reshape(3)
        if self._operator_origin is None:
            # X/Y translation comes from the detected center plus stereo depth,
            # so it can be a good origin even while rotation is in single-tag
            # fallback. The stability window rejects bad startup samples.
            trust_for_origin = depth_source == "stereo"
            if trust_for_origin:
                self._origin_candidates.append(p.copy())
            if len(self._origin_candidates) >= self._origin_warmup_min:
                pts = np.vstack(self._origin_candidates)
                spread = np.max(np.linalg.norm(pts - np.median(pts, axis=0), axis=1))
                if spread <= self._origin_max_spread_m:
                    self._operator_origin = np.median(pts, axis=0)
                    self.operator_filter.reset()
                    return p - self._operator_origin, "locked"

            # Until the origin is reliable, keep the top-down point near zero
            # instead of locking onto a bad early depth/pose sample.
            return np.zeros(3, dtype=np.float64), "warming"
        return p - self._operator_origin, "locked"

    def _sample_depth_m(
        self,
        depth_frame: Optional[np.ndarray],
        center_px: tuple[float, float],
    ) -> Optional[float]:
        if depth_frame is None:
            return None
        depth = np.asarray(depth_frame)
        if depth.size == 0:
            return None

        h, w = depth.shape[:2]
        u = int(round(float(center_px[0])))
        v = int(round(float(center_px[1])))
        half = max(0, self.depth_roi_px // 2)
        x0 = max(0, u - half)
        x1 = min(w, u + half + 1)
        y0 = max(0, v - half)
        y1 = min(h, v + half + 1)
        if x0 >= x1 or y0 >= y1:
            return None

        roi = depth[y0:y1, x0:x1].astype(np.float64).reshape(-1)
        roi = roi[np.isfinite(roi)]
        roi = roi[roi > 0.0]
        if roi.size == 0:
            return None

        # DepthAI depth frames are normally uint16 millimeters. If a float meter
        # frame is ever supplied, keep it as meters.
        if np.nanmedian(roi) > 50.0:
            roi = roi / 1000.0

        roi = roi[(roi >= self.depth_min_m) & (roi <= self.depth_max_m)]
        if roi.size == 0:
            return None
        return float(np.median(roi))

    def _solve_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
    ) -> Optional[tuple[np.ndarray, np.ndarray, float, str]]:
        candidates: list[tuple[np.ndarray, np.ndarray, float, str]] = []
        obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        is_planar = _points_are_coplanar(obj)

        if is_planar and hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE"):
            try:
                out = cv2.solvePnPGeneric(
                    obj,
                    img,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                ok, rvecs, tvecs = bool(out[0]), out[1], out[2]
                if ok:
                    for rvec, tvec in zip(rvecs, tvecs):
                        t = np.asarray(tvec, dtype=np.float64).reshape(3)
                        if t[2] <= 0:
                            continue
                        err = self._rms_reprojection(
                            obj, img, rvec, tvec,
                            camera_matrix, dist_coeffs,
                        )
                        candidates.append((rvec, tvec, err, "multi-tag"))
            except cv2.error:
                pass

        for flag_name in ("SOLVEPNP_SQPNP", "SOLVEPNP_EPNP"):
            if is_planar or not hasattr(cv2, flag_name):
                continue
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    obj,
                    img,
                    camera_matrix,
                    dist_coeffs,
                    flags=getattr(cv2, flag_name),
                )
                if ok and float(np.asarray(tvec).reshape(3)[2]) > 0:
                    err = self._rms_reprojection(
                        obj, img, rvec, tvec,
                        camera_matrix, dist_coeffs,
                    )
                    candidates.append((rvec, tvec, err, "multi-tag 3d"))
            except cv2.error:
                pass

        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj,
                img,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok and float(np.asarray(tvec).reshape(3)[2]) > 0:
                err = self._rms_reprojection(
                    obj, img, rvec, tvec,
                    camera_matrix, dist_coeffs,
                )
                candidates.append((rvec, tvec, err, "multi-tag" if is_planar else "multi-tag 3d"))
        except cv2.error:
            pass

        if not candidates:
            return None
        candidates.sort(key=self._candidate_score)
        return candidates[0]

    def _solve_single_tag_pose(
        self,
        detections: list,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
    ) -> Optional[tuple[np.ndarray, np.ndarray, float, str]]:
        if not self.single_tag_fallback:
            return None

        candidates: list[tuple[np.ndarray, np.ndarray, float, str]] = []
        for det in detections:
            obj = self.layout.object_corners(int(det.tag_id)).reshape(-1, 3)
            img = np.asarray(det.corners, dtype=np.float64).reshape(-1, 2)
            solved = self._solve_pose(obj, img, camera_matrix, dist_coeffs)
            if solved is None:
                continue
            rvec, tvec, err, _source = solved
            candidates.append((rvec, tvec, err, f"single-tag ID{int(det.tag_id)}"))

        if not candidates:
            return None
        candidates.sort(key=self._candidate_score)
        return candidates[0]

    def _candidate_score(self, candidate: tuple[np.ndarray, np.ndarray, float, str]) -> float:
        rvec, tvec, reproj_err, source = candidate
        score = float(reproj_err)
        if source.startswith("single"):
            score += 4.0

        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        rmat, _ = cv2.Rodrigues(rvec)
        center = _target_center_camera(rmat, t, self.layout.target_center_m)

        detector_center = getattr(self, "_detector_center_cam", None)
        if detector_center is not None:
            score += 50.0 * float(np.linalg.norm(center - detector_center))

        prev_center = getattr(self, "_prev_target_center_camera", None)
        if prev_center is None:
            return score

        trans_m = float(np.linalg.norm(center - prev_center))
        # Reprojection error still dominates. The translation term is based on
        # the target center so face handoffs are not mistaken for lateral jumps.
        # Rotation is deliberately not penalized here: when the cube turns from
        # one face to another, a large rotation can be the correct explanation.
        return score + 20.0 * trans_m

    def _update_omega(self, yaw_deg: float, timestamp: float) -> tuple[float, float]:
        yaw = float(yaw_deg)
        ts = float(timestamp)
        if self._prev_yaw_deg is None or self._prev_unwrapped_yaw_deg is None:
            self._prev_yaw_deg = yaw
            self._prev_unwrapped_yaw_deg = yaw
            self._prev_omega_ts = ts
            self._omega_deg_s = 0.0
            return 0.0, 0.0

        delta = wrapped_angle_delta_deg(yaw, self._prev_yaw_deg)
        unwrapped = self._prev_unwrapped_yaw_deg + delta
        if self._prev_omega_ts is not None:
            dt = ts - self._prev_omega_ts
            if dt > 1e-4:
                raw_omega = delta / dt
                self._omega_deg_s = 0.45 * raw_omega + 0.55 * self._omega_deg_s

        self._prev_yaw_deg = yaw
        self._prev_unwrapped_yaw_deg = unwrapped
        self._prev_omega_ts = ts
        return float(self._omega_deg_s), math.radians(float(self._omega_deg_s))

    @staticmethod
    def _rms_reprojection(
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
    ) -> float:
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        delta = projected - image_points.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


class TopDownView:
    MIN_SPAN = 0.30
    ZERO_EXTENT = 0.12
    PIXEL_MARGIN = 24

    def __init__(self, width: int, height: int, *, x_range: float, y_range: float):
        self.width = int(width)
        self.height = int(height)
        self.x_range = float(x_range)
        self.y_range = float(y_range)
        self.trail: deque[tuple[float, float]] = deque(maxlen=120)

    def reset(self) -> None:
        self.trail.clear()

    def render(
        self,
        marker_xy: Optional[tuple[float, float]],
        *,
        status: str,
        yaw_deg: Optional[float],
        update_trail: bool,
        mode: str,
    ) -> np.ndarray:
        bounds = self._bounds(marker_xy)
        img = np.full((self.height, self.width, 3), (24, 24, 30), dtype=np.uint8)
        self._draw_grid(img, bounds)

        if marker_xy is not None and update_trail:
            self.trail.append((float(marker_xy[0]), float(marker_xy[1])))

        for i in range(1, len(self.trail)):
            p1 = self._world_to_pixel(*self.trail[i - 1], bounds)
            p2 = self._world_to_pixel(*self.trail[i], bounds)
            a = i / max(1, len(self.trail))
            cv2.line(img, p1, p2, (int(50 * a), int(230 * a), int(90 * a)), 2)

        if marker_xy is not None:
            px, py = self._world_to_pixel(marker_xy[0], marker_xy[1], bounds)
            color = (0, 255, 0) if status == "GOOD" else (0, 190, 255)
            if status == "HOLD":
                color = (0, 165, 255)
            cv2.circle(img, (px, py), 8, color, -1)
            cv2.circle(img, (px, py), 15, color, 2)
            if yaw_deg is not None and math.isfinite(yaw_deg):
                rad = math.radians(yaw_deg)
                end = (
                    int(round(px + 26 * math.sin(rad))),
                    int(round(py - 26 * math.cos(rad))),
                )
                cv2.arrowedLine(img, (px, py), end, color, 2, tipLength=0.35)
            cv2.putText(
                img,
                (
                    f"X_abs={marker_xy[0]:+.3f}m  Y_abs={marker_xy[1]:+.3f}m"
                    if mode == "abs"
                    else f"dX={marker_xy[0]:+.3f}m  dY={marker_xy[1]:+.3f}m"
                ),
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )
        else:
            cv2.putText(
                img,
                f"{status}: NO POSE",
                (max(12, self.width // 2 - 95), self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 130, 255),
                2,
            )

        cv2.putText(
            img,
            (
                "TOP-DOWN ABS X/Y (smoothed stereo range)"
                if mode == "abs"
                else "TOP-DOWN REL dX/dY"
            ),
            (10, self.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 170, 180),
            1,
        )
        return img

    def _bounds(self, marker_xy: Optional[tuple[float, float]]):
        xs = [0.0, -self.ZERO_EXTENT, self.ZERO_EXTENT]
        ys = [0.0, -self.ZERO_EXTENT, self.ZERO_EXTENT]
        xs.extend([-self.x_range, self.x_range])
        ys.extend([-self.y_range, self.y_range])
        if marker_xy is not None:
            xs.append(float(marker_xy[0]))
            ys.append(float(marker_xy[1]))
        for x, y in self.trail:
            xs.append(float(x))
            ys.append(float(y))
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max - x_min < self.MIN_SPAN:
            c = 0.5 * (x_min + x_max)
            x_min, x_max = c - 0.5 * self.MIN_SPAN, c + 0.5 * self.MIN_SPAN
        if y_max - y_min < self.MIN_SPAN:
            c = 0.5 * (y_min + y_max)
            y_min, y_max = c - 0.5 * self.MIN_SPAN, c + 0.5 * self.MIN_SPAN
        x_pad = max(0.05, 0.12 * (x_max - x_min))
        y_pad = max(0.05, 0.12 * (y_max - y_min))
        return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad

    def _world_to_pixel(self, x: float, y: float, bounds) -> tuple[int, int]:
        x_min, x_max, y_min, y_max = bounds
        m = self.PIXEL_MARGIN
        pw = max(1, self.width - 2 * m)
        ph = max(1, self.height - 2 * m)
        px = int(round(m + (float(x) - x_min) / max(1e-9, x_max - x_min) * pw))
        py = int(round(m + (1.0 - (float(y) - y_min) / max(1e-9, y_max - y_min)) * ph))
        return (
            int(np.clip(px, m, self.width - m - 1)),
            int(np.clip(py, m, self.height - m - 1)),
        )

    def _draw_grid(self, img: np.ndarray, bounds) -> None:
        x_min, x_max, y_min, y_max = bounds
        grid_color = (56, 56, 66)
        for x in np.arange(math.floor(x_min * 4) / 4, x_max + 0.01, 0.25):
            px, _ = self._world_to_pixel(x, 0.0, bounds)
            cv2.line(img, (px, 0), (px, self.height), grid_color, 1)
        for y in np.arange(math.floor(y_min * 4) / 4, y_max + 0.01, 0.25):
            _, py = self._world_to_pixel(0.0, y, bounds)
            cv2.line(img, (0, py), (self.width, py), grid_color, 1)

        ox, oy = self._world_to_pixel(0.0, 0.0, bounds)
        axis_color = (90, 110, 140)
        cv2.line(img, (ox, 0), (ox, self.height), axis_color, 1)
        cv2.line(img, (0, oy), (self.width, oy), axis_color, 1)
        cv2.drawMarker(img, (ox, oy), (215, 215, 225), cv2.MARKER_CROSS, 12, 2)


def _drain_queue(queue):
    msg = None
    while True:
        item = queue.tryGet()
        if item is None:
            break
        msg = item
    return msg


def _camera_socket(name: str):
    try:
        return getattr(dai.CameraBoardSocket, name)
    except AttributeError as exc:
        valid = [n for n in dir(dai.CameraBoardSocket) if n.startswith("CAM_")]
        raise ValueError(f"unknown camera socket {name!r}; valid examples: {valid}") from exc


def _scaled_camera_matrix(camera_matrix: np.ndarray, sx: float, sy: float) -> np.ndarray:
    k = np.asarray(camera_matrix, dtype=np.float64).copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    return k


def _undistort_frame_cached(
    gray: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    cache: dict,
) -> np.ndarray:
    key = (gray.shape[:2], camera_matrix.tobytes(), dist_coeffs.tobytes())
    if cache.get("key") != key:
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
    return cv2.remap(gray, cache["map1"], cache["map2"], cv2.INTER_LINEAR)


def _frame_array(msg) -> np.ndarray:
    if hasattr(msg, "getFrame"):
        return msg.getFrame()
    return msg.getCvFrame()


def _draw_tags(
    image: np.ndarray,
    detections: list,
    *,
    scale_x: float,
    scale_y: float,
    status: str,
) -> None:
    color = (0, 255, 0) if status == "GOOD" else (0, 190, 255)
    if status in ("BAD", "REACQ"):
        color = (0, 130, 255)
    for det in detections:
        corners = np.asarray(det.corners, dtype=np.float32).copy()
        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y
        pts = corners.astype(np.int32)
        cv2.polylines(image, [pts], True, color, 2)
        c = corners.mean(axis=0)
        cv2.circle(image, (int(c[0]), int(c[1])), 4, color, -1)
        cv2.putText(
            image,
            f"ID{int(det.tag_id)}",
            (int(c[0]) + 7, int(c[1]) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def _draw_axes(
    image: np.ndarray,
    pose: PoseEstimate,
    pose_camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_len_m: float,
    target_center_m: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> None:
    center = np.asarray(target_center_m, dtype=np.float64).reshape(3)
    obj_pts = np.vstack(
        [
            center,
            center + np.array([axis_len_m, 0.0, 0.0], dtype=np.float64),
            center + np.array([0.0, axis_len_m, 0.0], dtype=np.float64),
            center + np.array([0.0, 0.0, axis_len_m], dtype=np.float64),
        ]
    )
    pts, _ = cv2.projectPoints(
        obj_pts,
        pose.rvec,
        pose.tvec,
        pose_camera_matrix,
        dist_coeffs,
    )
    pts = pts.reshape(-1, 2)
    pts[:, 0] *= float(scale_x)
    pts[:, 1] *= float(scale_y)
    pts = pts.astype(int)
    o = tuple(pts[0])
    cv2.arrowedLine(image, o, tuple(pts[1]), (0, 0, 255), 3, tipLength=0.25)
    cv2.arrowedLine(image, o, tuple(pts[2]), (0, 255, 0), 3, tipLength=0.25)
    cv2.arrowedLine(image, o, tuple(pts[3]), (255, 0, 0), 3, tipLength=0.25)
    cv2.putText(image, "X", tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(image, "Y", tuple(pts[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(image, "Z", tuple(pts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    center_color = (0, 255, 0) if pose.center_err_mm < 5.0 else (0, 255, 255)
    cv2.circle(image, o, 9, center_color, -1)
    cv2.circle(image, o, 15, center_color, 2)
    cv2.putText(
        image,
        "CENTER",
        (o[0] + 12, o[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        center_color,
        2,
    )


def _draw_text_panel(
    image: np.ndarray,
    result: Optional[FrameResult],
    *,
    detect_fps: float,
    stream_fps: float,
    topdown_mode: str,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"det {detect_fps:.0f} FPS  stream {stream_fps:.0f} FPS",
    ]
    color = (210, 210, 210)
    if result is None:
        lines.append("No frames yet")
        status = "WAIT"
    else:
        status = result.status
        lines.append(f"{status}: {result.message}")
        if result.pose is not None:
            p = result.pose
            abs_t = p.operator_abs_smooth
            rel_t = p.operator_t_smooth
            a = p.rel_euler_smooth_deg
            lines.append(
                f"absolute smoothed: X={abs_t[0]:+.3f} Y(range)={abs_t[1]:+.3f} Z={abs_t[2]:+.3f} m"
            )
            lines.append(
                f"relative: dX={rel_t[0]:+.3f} dY={rel_t[1]:+.3f} dZ={rel_t[2]:+.3f} m"
            )
            lines.append(
                f"rotation: roll={a[0]:+.1f} pitch={a[1]:+.1f} yaw={a[2]:+.1f} deg"
            )
            lines.append(
                f"omega={p.omega_rad_s:+.3f} rad/s ({p.omega_deg_s:+.1f} deg/s)"
            )
            lines.append(
                f"depth={p.depth_source} {p.depth_m:.3f}m  tags={p.tag_ids} "
                f"err={p.reproj_error_px:.1f}px"
            )
            origin_s = f"  origin={p.origin_status}" if topdown_mode == "relative" else ""
            face_s = (
                f"  face={list(p.primary_face_ids)}"
                if p.primary_face_ids
                else ""
            )
            lines.append(
                f"pose={p.pose_source}  center={p.center_source}  "
                f"center_err={p.center_err_mm:.1f}mm{face_s}{origin_s}"
            )

    if status == "GOOD":
        color = (0, 255, 0)
    elif status == "WARN":
        color = (0, 190, 255)
    elif status == "HOLD":
        color = (0, 165, 255)
    elif status in ("BAD", "REACQ"):
        color = (0, 130, 255)

    x0, y0 = 12, 16
    line_h = 24
    panel_w = 610
    panel_h = line_h * len(lines) + 16
    cv2.rectangle(image, (x0 - 6, y0 - 14), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x0, y0 + i * line_h),
            font,
            0.62,
            color if i > 0 else (220, 220, 220),
            2 if i <= 1 else 1,
        )


def compose_view(
    gray: np.ndarray,
    result: Optional[FrameResult],
    topdown: TopDownView,
    *,
    pose_camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    target_center_m: np.ndarray,
    scale_x: float,
    scale_y: float,
    axis_len_m: float,
    detect_fps: float,
    stream_fps: float,
    topdown_mode: str,
) -> np.ndarray:
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    status = result.status if result is not None else "WAIT"
    if result is not None:
        _draw_tags(view, result.detections, scale_x=scale_x, scale_y=scale_y, status=status)
        if result.pose is not None:
            _draw_axes(
                view,
                result.pose,
                pose_camera_matrix,
                dist_coeffs,
                axis_len_m,
                target_center_m,
                scale_x,
                scale_y,
            )
    _draw_text_panel(
        view,
        result,
        detect_fps=detect_fps,
        stream_fps=stream_fps,
        topdown_mode=topdown_mode,
    )

    marker_xy = None
    yaw = None
    update_trail = False
    if result is not None and result.pose is not None:
        pos = (
            result.pose.operator_abs_smooth
            if topdown_mode == "abs"
            else result.pose.operator_t_smooth
        )
        marker_xy = (float(pos[0]), float(pos[1]))
        yaw = float(result.pose.rel_euler_smooth_deg[2])
        update_trail = result.status in ("GOOD", "WARN")

    td = topdown.render(
        marker_xy,
        status=status,
        yaw_deg=yaw,
        update_trail=update_trail,
        mode=topdown_mode,
    )
    if td.shape[0] != view.shape[0]:
        td = cv2.resize(td, (view.shape[0], view.shape[0]))
    return np.hstack([view, td])


def build_arg_parser() -> argparse.ArgumentParser:
    default_layout = Path(__file__).with_name("tag_layout.json")
    parser = argparse.ArgumentParser(description="Standalone OAK AprilTag wall tracker")
    parser.add_argument("--ip", default=os.environ.get("OAK_IP", "192.168.0.153"))
    parser.add_argument("--camera-socket", default="CAM_B")
    parser.add_argument("--layout", type=Path, default=default_layout)
    parser.add_argument("--tag-family", default=None)
    parser.add_argument("--tag-size", type=float, default=None)
    parser.add_argument("--detect-width", type=int, default=640)
    parser.add_argument("--detect-height", type=int, default=400)
    parser.add_argument("--detect-fps", type=int, default=120)
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--display-height", type=int, default=600)
    parser.add_argument("--display-fps", type=int, default=10)
    parser.add_argument("--quad-decimate", type=float, default=1.5)
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--decision-margin-min", type=float, default=8.0)
    parser.add_argument("--min-pose-tags", type=int, default=2)
    parser.add_argument("--max-reproj-error-px", type=float, default=8.0)
    parser.add_argument("--pose-alpha", type=float, default=0.35)
    parser.add_argument("--angle-alpha", type=float, default=0.35)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--max-translation-step-m", type=float, default=0.020)
    parser.add_argument("--max-angle-step-deg", type=float, default=4.0)
    parser.add_argument("--axis-len", type=float, default=0.12)
    parser.add_argument("--stream-port", type=int, default=8090)
    parser.add_argument("--jpeg-quality", type=int, default=68)
    parser.add_argument("--topdown-size", type=int, default=360)
    parser.add_argument("--topdown-mode", choices=("abs", "relative"), default="abs",
                        help="Top-down plot mode: abs range X/Y or origin-relative dX/dY")
    parser.add_argument("--x-range", type=float, default=0.75)
    parser.add_argument("--y-range", type=float, default=0.75)
    parser.add_argument("--z-range", type=float, default=None,
                        help="Deprecated alias for --y-range")
    parser.add_argument("--hold-sec", type=float, default=0.35)
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable OAK stereo depth and use PnP translation only")
    parser.add_argument("--depth-roi-px", type=int, default=21,
                        help="Median depth sampling window around target center")
    parser.add_argument("--depth-min-m", type=float, default=0.20)
    parser.add_argument("--depth-max-m", type=float, default=8.0)
    parser.add_argument("--depth-right-socket", default="CAM_C")
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--max-depth-age-sec", type=float, default=0.15,
                        help="Ignore stereo frames older than this relative to detection")
    parser.add_argument("--graph-deadband-m", type=float, default=0.008,
                        help="Ignore graph position jitter smaller than this distance")
    parser.add_argument("--no-single-tag-fallback", action="store_true",
                        help="Require the configured multi-tag layout for pose")
    parser.add_argument("--high-fps", action="store_true",
                        help="Prioritize AprilTag detection over stream smoothness")
    parser.add_argument("--stream-from-detect-only", action="store_true",
                        help="Do not request a separate camera display output")
    parser.add_argument("--no-stream", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    layout = TagLayout.load(args.layout, tag_size_override=args.tag_size)
    if args.tag_family:
        layout.tag_family = args.tag_family
    if args.ip:
        os.environ["DEPTHAI_DEVICE_NAME"] = args.ip
    if args.z_range is not None:
        args.y_range = float(args.z_range)
    if args.high_fps:
        args.detect_fps = max(args.detect_fps, 160)
        args.display_fps = min(args.display_fps, 8)
        args.display_width = min(args.display_width, 800)
        args.display_height = min(args.display_height, 500)
        args.topdown_size = min(args.topdown_size, 300)
        args.jpeg_quality = min(args.jpeg_quality, 60)
        args.stream_from_detect_only = True

    print("[MAIN] Standalone wall tracker")
    print(f"[MAIN] OAK IP: {args.ip or 'auto-discover'}")
    print(f"[MAIN] Layout: {args.layout} ({layout.name})")
    print(f"[MAIN] Tags: {sorted(layout.valid_ids)}  size={layout.tag_size_m:.3f} m")
    print(
        f"[MAIN] Detection: {args.detect_width}x{args.detect_height} "
        f"@ {args.detect_fps} FPS grayscale"
    )
    if not args.no_stream:
        stream_source = "detection frame" if args.stream_from_detect_only else "display output"
        print(
            f"[MAIN] Stream: {args.display_width}x{args.display_height} "
            f"@ {args.display_fps} FPS q={args.jpeg_quality} source={stream_source}"
        )
        if args.high_fps:
            print("[MAIN] High-FPS mode: stream work is throttled for tracking priority")

    estimator = RigidTagPoseEstimator(
        layout,
        decision_margin_min=args.decision_margin_min,
        quad_decimate=args.quad_decimate,
        nthreads=args.nthreads,
        min_pose_tags=args.min_pose_tags,
        max_reproj_error_px=args.max_reproj_error_px,
        pose_alpha=args.pose_alpha,
        angle_alpha=args.angle_alpha,
        median_window=args.median_window,
        max_translation_step_m=args.max_translation_step_m,
        max_angle_step_deg=args.max_angle_step_deg,
        depth_roi_px=args.depth_roi_px,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        single_tag_fallback=not args.no_single_tag_fallback,
        graph_deadband_m=args.graph_deadband_m,
    )

    server = None
    if not args.no_stream:
        server = start_stream_server(args.stream_port)
        if server is not None:
            print(f"[MAIN] Web stream: http://{local_lan_ip()}:{args.stream_port}/")

    topdown = TopDownView(
        args.topdown_size,
        args.topdown_size,
        x_range=args.x_range,
        y_range=args.y_range,
    )

    socket_name = args.camera_socket
    socket_id = _camera_socket(socket_name)

    latest_result: Optional[FrameResult] = None
    latest_display_result: Optional[FrameResult] = None
    last_pose_result: Optional[FrameResult] = None
    last_pose_ts: Optional[float] = None
    latest_detect_gray: Optional[np.ndarray] = None
    latest_depth_frame: Optional[np.ndarray] = None
    latest_depth_ts: Optional[float] = None
    latest_detect_ts = 0.0
    detect_fps_counter = 0
    stream_fps_counter = 0
    detect_fps = 0.0
    stream_fps = 0.0
    stream_period = 1.0 / max(1.0, float(args.display_fps))
    next_stream_time = 0.0
    fps_time = time.monotonic()
    stream_undistort_cache: dict = {}

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

        display_q = None
        depth_q = None
        depth_enabled = False
        if not args.no_depth:
            try:
                right_socket_id = _camera_socket(args.depth_right_socket)
                right_cam = pipeline.create(dai.node.Camera).build(right_socket_id)
                stereo = pipeline.create(dai.node.StereoDepth)
                if hasattr(stereo, "setDefaultProfilePreset"):
                    stereo.setDefaultProfilePreset(
                        dai.node.StereoDepth.PresetMode.DENSITY
                    )
                if hasattr(stereo, "setDepthAlign"):
                    stereo.setDepthAlign(socket_id)
                if hasattr(stereo, "setOutputSize"):
                    stereo.setOutputSize(args.detect_width, args.detect_height)

                right_depth_out = right_cam.requestOutput(
                    (args.detect_width, args.detect_height),
                    dai.ImgFrame.Type.GRAY8,
                    fps=args.depth_fps,
                )
                # Reuse the detection stream as the left stereo input. Requesting
                # a second left-camera stream can starve the host detection queue
                # on some OAK/DepthAI combinations.
                detect_out.link(stereo.left)
                right_depth_out.link(stereo.right)
                depth_q = stereo.depth.createOutputQueue()
                depth_q.setMaxSize(1)
                depth_q.setBlocking(False)
                depth_enabled = True
            except Exception as exc:
                print(f"[DEPTH] Disabled: could not start stereo branch ({exc})")
        if server is not None and not args.stream_from_detect_only:
            display_out = cam.requestOutput(
                (args.display_width, args.display_height),
                dai.ImgFrame.Type.GRAY8,
                fps=args.display_fps,
            )
            display_q = display_out.createOutputQueue()
            display_q.setMaxSize(1)
            display_q.setBlocking(False)

        pipeline.start()
        device = pipeline.getDefaultDevice()
        calib = device.readCalibration()
        detect_k = np.array(
            calib.getCameraIntrinsics(
                socket_id, args.detect_width, args.detect_height
            ),
            dtype=np.float64,
        )
        dist = np.array(calib.getDistortionCoefficients(socket_id), dtype=np.float64)
        dist = dist.reshape(-1, 1)
        display_k = np.array(
            calib.getCameraIntrinsics(
                socket_id, args.display_width, args.display_height
            ),
            dtype=np.float64,
        )
        if depth_enabled:
            print(
                f"[DEPTH] Stereo enabled: left={socket_name} "
                f"right={args.depth_right_socket} roi={args.depth_roi_px}px"
            )
        else:
            print("[DEPTH] Stereo disabled; using PnP depth fallback")

        print("[MAIN] Running. Ctrl+C to stop.\n")
        try:
            while pipeline.isRunning():
                got_work = False
                if depth_q is not None:
                    depth_msg = _drain_queue(depth_q)
                    if depth_msg is not None:
                        got_work = True
                        latest_depth_frame = _frame_array(depth_msg)
                        latest_depth_ts = depth_msg.getTimestamp().total_seconds()
                while True:
                    msg = detect_q.tryGet()
                    if msg is None:
                        break
                    got_work = True
                    gray = msg.getCvFrame()
                    ts = msg.getTimestamp().total_seconds()
                    latest_detect_ts = ts
                    depth_frame_for_pose = None
                    if (
                        depth_enabled
                        and latest_depth_frame is not None
                        and latest_depth_ts is not None
                        and abs(ts - latest_depth_ts) <= args.max_depth_age_sec
                    ):
                        depth_frame_for_pose = latest_depth_frame
                    latest_result = estimator.process(
                        gray, detect_k, dist, ts,
                        depth_frame=depth_frame_for_pose,
                    )
                    latest_detect_gray = estimator._undistort_gray(gray)
                    if latest_result.pose is not None:
                        latest_display_result = latest_result
                        last_pose_result = latest_result
                        last_pose_ts = ts
                    elif (
                        last_pose_result is not None
                        and last_pose_result.pose is not None
                        and last_pose_ts is not None
                        and (ts - last_pose_ts) <= args.hold_sec
                    ):
                        latest_display_result = FrameResult(
                            detections=latest_result.detections,
                            pose=last_pose_result.pose,
                            status="HOLD",
                            message=(
                                f"holding {ts - last_pose_ts:.2f}s; "
                                f"{latest_result.message}"
                            ),
                            timestamp=ts,
                        )
                    else:
                        latest_display_result = latest_result
                    detect_fps_counter += 1

                loop_now = time.monotonic()
                if server is not None and loop_now >= next_stream_time:
                    next_stream_time = loop_now + stream_period
                    frame_msg = _drain_queue(display_q) if display_q is not None else None
                    if frame_msg is not None:
                        got_work = True
                        display_gray = _undistort_frame_cached(
                            frame_msg.getCvFrame(),
                            display_k,
                            dist,
                            stream_undistort_cache,
                        )
                        k_for_view = display_k
                        sx = args.display_width / args.detect_width
                        sy = args.display_height / args.detect_height
                    elif latest_detect_gray is not None:
                        display_gray = cv2.resize(
                            latest_detect_gray,
                            (args.display_width, args.display_height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        k_for_view = _scaled_camera_matrix(
                            detect_k,
                            args.display_width / args.detect_width,
                            args.display_height / args.detect_height,
                        )
                        sx = args.display_width / args.detect_width
                        sy = args.display_height / args.detect_height
                    else:
                        display_gray = None

                    if display_gray is not None:
                        composed = compose_view(
                            display_gray,
                            latest_display_result,
                            topdown,
                            pose_camera_matrix=detect_k,
                            dist_coeffs=None,
                            target_center_m=layout.target_center_m,
                            scale_x=sx,
                            scale_y=sy,
                            axis_len_m=args.axis_len,
                            detect_fps=detect_fps,
                            stream_fps=stream_fps,
                            topdown_mode=args.topdown_mode,
                        )
                        stream.update(composed, quality=args.jpeg_quality)
                        stream_fps_counter += 1

                now = time.monotonic()
                if now - fps_time >= 1.0:
                    dt = now - fps_time
                    detect_fps = detect_fps_counter / dt
                    stream_fps = stream_fps_counter / dt
                    detect_fps_counter = 0
                    stream_fps_counter = 0
                    fps_time = now
                    if latest_display_result is not None:
                        msg = latest_display_result.message
                        if latest_display_result.pose is not None:
                            abs_t = latest_display_result.pose.operator_abs_smooth
                            rel_t = latest_display_result.pose.operator_t_smooth
                            a = latest_display_result.pose.rel_euler_smooth_deg
                            p = latest_display_result.pose
                            origin_log = (
                                f" origin={p.origin_status}"
                                if args.topdown_mode == "relative"
                                else ""
                            )
                            msg = (
                                f"{msg} abs=({abs_t[0]:+.3f},{abs_t[1]:+.3f},{abs_t[2]:+.3f}) "
                                f"rel=({rel_t[0]:+.3f},{rel_t[1]:+.3f},{rel_t[2]:+.3f}) "
                                f"depth={p.depth_source} "
                                f"roll={a[0]:+.1f} pitch={a[1]:+.1f} "
                                f"yaw={a[2]:+.1f} omega={p.omega_rad_s:+.3f}rad/s"
                                f"{origin_log}"
                            )
                        print(
                            f"[{latest_display_result.status}] det={detect_fps:.0f} "
                            f"stream={stream_fps:.0f} {msg}"
                        )

                if not got_work:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print("\n[MAIN] Stopping.")

    if server is not None:
        server.shutdown()
    _ = latest_detect_ts
    print("[MAIN] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
