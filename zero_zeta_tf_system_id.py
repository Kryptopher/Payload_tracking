#!/usr/bin/env python3
"""
Live zero-zeta system ID from a TFMessage payload swing signal.

This script is intentionally standalone so it can be copied to another ROS2
workspace. It subscribes to a TFMessage topic, extracts one translation
component, and fits

    y(t) = c0 + c1*t + a*cos(omega*t) + b*sin(omega*t)

over a grid of candidate omega values. Damping is assumed zero, so the output is
omega only; zeta is reported as 0.0 by design.

Typical use:

    python3 zero_zeta_tf_system_id.py \
      --topic /apriltags/base_cam/rgb/tf \
      --axis x \
      --input-scale 1000 \
      --child-frame-id payload_center

The output topic is a Float64MultiArray with fields:

    time,omega_rad_s,zeta,shaper_T_s,osc_period_s,freq_hz,
    amplitude_mm,rmse_mm,nrmse,p2p_mm,num_samples,sample_rate_hz
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from tf2_msgs.msg import TFMessage


ESTIMATE_FIELDS = (
    'time',
    'omega_rad_s',
    'zeta',
    'shaper_T_s',
    'osc_period_s',
    'freq_hz',
    'amplitude_mm',
    'rmse_mm',
    'nrmse',
    'p2p_mm',
    'num_samples',
    'sample_rate_hz',
)


@dataclass
class Sample:
    t: float
    y_mm: float


@dataclass
class ZeroZetaEstimate:
    fit_time_s: float
    omega_rad_s: float
    shaper_T_s: float
    osc_period_s: float
    freq_hz: float
    amplitude_mm: float
    rmse_mm: float
    nrmse: float
    p2p_mm: float
    num_samples: int
    sample_rate_hz: float
    condition: float


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def robust_sample_rate(times: np.ndarray) -> float:
    if len(times) < 2:
        return float('nan')
    dt = np.diff(np.sort(times))
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return float('nan')
    return 1.0 / float(np.median(dt))


def fit_zero_zeta_grid(
    times: np.ndarray,
    values_mm: np.ndarray,
    half_period_min_s: float,
    half_period_max_s: float,
    grid_count: int,
    min_amp_mm: float,
    max_nrmse: float,
    min_samples_per_cycle: float,
) -> Optional[ZeroZetaEstimate]:
    """Fit zero-damping sinusoid plus offset and linear drift.

    The "shaper_T_s" output is the input-shaper impulse spacing:

        T = pi / omega

    The physical oscillation period is:

        P = 2*pi / omega = 2*T
    """
    if len(times) < 4:
        return None
    order = np.argsort(times)
    times = np.asarray(times[order], dtype=float)
    values_mm = np.asarray(values_mm[order], dtype=float)
    finite = np.isfinite(times) & np.isfinite(values_mm)
    times = times[finite]
    values_mm = values_mm[finite]
    if len(times) < 4:
        return None

    sample_rate_hz = robust_sample_rate(times)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        return None

    p2p_mm = float(np.max(values_mm) - np.min(values_mm))
    if p2p_mm <= 0.0:
        return None

    tc = times - float(times[0])
    y = values_mm
    half_periods = np.linspace(
        half_period_min_s, half_period_max_s, max(5, int(grid_count)))

    # Reject candidates that are too fast for the observed camera rate.
    min_full_period = min_samples_per_cycle / sample_rate_hz
    half_periods = half_periods[(2.0 * half_periods) >= min_full_period]
    if len(half_periods) == 0:
        return None

    best = None
    for shaper_T_s in half_periods:
        omega = math.pi / float(shaper_T_s)
        phase = omega * tc
        design = np.column_stack((
            np.ones_like(tc),
            tc,
            np.cos(phase),
            np.sin(phase),
        ))
        try:
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            pred = design @ coef
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            amp = float(math.hypot(float(coef[2]), float(coef[3])))
            nrmse = rmse / max(amp, 1.0e-9)
            condition = float(np.linalg.cond(design))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            continue
        if not (
            math.isfinite(rmse)
            and math.isfinite(amp)
            and math.isfinite(nrmse)
            and math.isfinite(condition)
        ):
            continue
        if amp < min_amp_mm or nrmse > max_nrmse:
            continue
        if best is None or nrmse < best.nrmse:
            best = ZeroZetaEstimate(
                fit_time_s=float(times[-1]),
                omega_rad_s=float(omega),
                shaper_T_s=float(shaper_T_s),
                osc_period_s=float(2.0 * shaper_T_s),
                freq_hz=float(omega / (2.0 * math.pi)),
                amplitude_mm=amp,
                rmse_mm=rmse,
                nrmse=nrmse,
                p2p_mm=p2p_mm,
                num_samples=int(len(times)),
                sample_rate_hz=float(sample_rate_hz),
                condition=condition,
            )
    return best


class ZeroZetaTfSystemId(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('zero_zeta_tf_system_id')
        self.args = args
        self.samples: list[Sample] = []
        self.t0: Optional[float] = None
        self.last_update_wall = 0.0
        self.last_print_wall = 0.0
        self.multiple_transform_warning_time = 0.0

        self.pub = self.create_publisher(
            Float64MultiArray, args.estimate_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(
            TFMessage, args.topic, self.on_tf, qos_profile_sensor_data)

        self.csv_file = None
        self.csv_writer = None
        if args.log_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.log_csv)), exist_ok=True)
            self.csv_file = open(args.log_csv, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(ESTIMATE_FIELDS + ('condition',))

        self.get_logger().info(
            f'Zero-zeta TF system ID listening on {args.topic}; axis={args.axis}; '
            f'input_scale={args.input_scale}; output={args.estimate_topic}')
        self.get_logger().info(
            f'Search shaper T in [{args.half_period_min_s:.3f}, '
            f'{args.half_period_max_s:.3f}] s, history={args.history_s:.1f}s, '
            f'min_samples={args.min_samples}')
        if args.child_frame_id or args.parent_frame_id:
            self.get_logger().info(
                f'TF filter parent={args.parent_frame_id or "*"} '
                f'child={args.child_frame_id or "*"}')

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()

    def select_transform(self, msg: TFMessage):
        candidates = []
        for tf in msg.transforms:
            if self.args.child_frame_id and tf.child_frame_id != self.args.child_frame_id:
                continue
            if self.args.parent_frame_id and tf.header.frame_id != self.args.parent_frame_id:
                continue
            candidates.append(tf)
        if not candidates:
            return None
        if len(candidates) > 1 and not (self.args.child_frame_id or self.args.parent_frame_id):
            now = time.monotonic()
            if now - self.multiple_transform_warning_time > 5.0:
                self.multiple_transform_warning_time = now
                names = ', '.join(tf.child_frame_id for tf in candidates[:6])
                self.get_logger().warn(
                    f'{len(candidates)} transforms in message; using first ({names}). '
                    'Set --child-frame-id for repeatable ID.')
        return candidates[0]

    def extract_value_mm(self, tf) -> float:
        tr = tf.transform.translation
        sx = float(tr.x) * self.args.input_scale
        sy = float(tr.y) * self.args.input_scale
        sz = float(tr.z) * self.args.input_scale
        if self.args.axis == 'x':
            return sx
        if self.args.axis == 'y':
            return sy
        if self.args.axis == 'z':
            return sz
        if self.args.axis == 'xy':
            return math.hypot(sx, sy)
        raise ValueError(f'Unknown axis {self.args.axis}')

    def on_tf(self, msg: TFMessage):
        tf = self.select_transform(msg)
        if tf is None:
            return

        stamp_s = stamp_to_sec(tf.header.stamp)
        if stamp_s <= 0.0:
            stamp_s = self.get_clock().now().nanoseconds * 1.0e-9
        if self.t0 is None:
            self.t0 = stamp_s
        t = stamp_s - self.t0
        if t < 0.0:
            return

        y_mm = self.extract_value_mm(tf)
        self.samples.append(Sample(t=t, y_mm=y_mm))
        keep_after = t - self.args.history_s
        self.samples = [s for s in self.samples if s.t >= keep_after]

        now_wall = time.monotonic()
        if t < self.args.warmup_s:
            return
        if now_wall - self.last_update_wall < self.args.update_period_s:
            return
        self.last_update_wall = now_wall

        if len(self.samples) < self.args.min_samples:
            return

        times = np.array([s.t for s in self.samples], dtype=float)
        values = np.array([s.y_mm for s in self.samples], dtype=float)
        p2p = float(np.max(values) - np.min(values))
        if p2p < self.args.min_p2p_mm:
            return

        est = fit_zero_zeta_grid(
            times=times,
            values_mm=values,
            half_period_min_s=self.args.half_period_min_s,
            half_period_max_s=self.args.half_period_max_s,
            grid_count=self.args.grid_count,
            min_amp_mm=self.args.min_amp_mm,
            max_nrmse=self.args.max_nrmse,
            min_samples_per_cycle=self.args.min_samples_per_cycle,
        )
        if est is None:
            return

        self.publish_estimate(est)
        if now_wall - self.last_print_wall >= self.args.print_period_s:
            self.last_print_wall = now_wall
            self.print_estimate(est)

    def publish_estimate(self, est: ZeroZetaEstimate):
        msg = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label = ','.join(ESTIMATE_FIELDS)
        dim.size = len(ESTIMATE_FIELDS)
        dim.stride = len(ESTIMATE_FIELDS)
        msg.layout.dim.append(dim)
        msg.data = [
            est.fit_time_s,
            est.omega_rad_s,
            0.0,
            est.shaper_T_s,
            est.osc_period_s,
            est.freq_hz,
            est.amplitude_mm,
            est.rmse_mm,
            est.nrmse,
            est.p2p_mm,
            float(est.num_samples),
            est.sample_rate_hz,
        ]
        self.pub.publish(msg)
        if self.csv_writer is not None:
            self.csv_writer.writerow(msg.data + [est.condition])
            self.csv_file.flush()

    def print_estimate(self, est: ZeroZetaEstimate):
        samples_per_cycle = est.sample_rate_hz * est.osc_period_s
        warning = ''
        if samples_per_cycle < self.args.warn_samples_per_cycle:
            warning = (
                f' WARNING low sampling: {samples_per_cycle:.1f} samples/cycle')
        self.get_logger().info(
            f'omega={est.omega_rad_s:.4f} rad/s, '
            f'T_shaper={est.shaper_T_s:.3f}s, '
            f'period={est.osc_period_s:.3f}s, amp={est.amplitude_mm:.2f}mm, '
            f'nrmse={est.nrmse:.3f}, p2p={est.p2p_mm:.2f}mm, '
            f'n={est.num_samples}, fs={est.sample_rate_hz:.2f}Hz{warning}')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Zero-zeta sinusoid system ID from TFMessage translation.')
    parser.add_argument('--topic', default='/apriltags/base_cam/rgb/tf')
    parser.add_argument('--estimate-topic', default='/payload/zero_zeta_id')
    parser.add_argument('--child-frame-id', default='')
    parser.add_argument('--parent-frame-id', default='')
    parser.add_argument(
        '--axis',
        choices=('x', 'y', 'z', 'xy'),
        default='x',
        help='Translation component to fit. Use xy for planar magnitude.')
    parser.add_argument(
        '--input-scale',
        type=float,
        default=1000.0,
        help='Scale TF translation to millimeters. TF normally uses meters, so default is 1000.')
    parser.add_argument('--history-s', type=float, default=20.0)
    parser.add_argument('--warmup-s', type=float, default=4.0)
    parser.add_argument('--update-period-s', type=float, default=1.0)
    parser.add_argument('--print-period-s', type=float, default=1.0)
    parser.add_argument(
        '--half-period-min-s',
        type=float,
        default=0.50,
        help='Minimum shaper T = pi/omega, not the full oscillation period.')
    parser.add_argument(
        '--half-period-max-s',
        type=float,
        default=4.00,
        help='Maximum shaper T = pi/omega, not the full oscillation period.')
    parser.add_argument('--grid-count', type=int, default=350)
    parser.add_argument('--min-samples', type=int, default=18)
    parser.add_argument('--min-p2p-mm', type=float, default=4.0)
    parser.add_argument('--min-amp-mm', type=float, default=1.0)
    parser.add_argument('--max-nrmse', type=float, default=1.5)
    parser.add_argument(
        '--min-samples-per-cycle',
        type=float,
        default=4.0,
        help='Reject candidate frequencies faster than this sampling density.')
    parser.add_argument(
        '--warn-samples-per-cycle',
        type=float,
        default=6.0,
        help='Print a warning below this accepted sampling density.')
    parser.add_argument('--log-csv', default='')
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.history_s <= 0.0:
        raise ValueError('--history-s must be positive')
    if args.warmup_s < 0.0:
        raise ValueError('--warmup-s must be nonnegative')
    if args.update_period_s <= 0.0:
        raise ValueError('--update-period-s must be positive')
    if args.half_period_min_s <= 0.0 or args.half_period_max_s <= args.half_period_min_s:
        raise ValueError('Need 0 < --half-period-min-s < --half-period-max-s')
    if args.grid_count < 5:
        raise ValueError('--grid-count must be at least 5')
    if args.min_samples < 4:
        raise ValueError('--min-samples must be at least 4')
    if args.input_scale == 0.0:
        raise ValueError('--input-scale must be nonzero')


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    rclpy.init()
    node = ZeroZetaTfSystemId(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
