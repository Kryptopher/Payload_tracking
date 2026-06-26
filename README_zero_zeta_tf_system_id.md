# Zero-Zeta TF System ID

`zero_zeta_tf_system_id.py` is a standalone ROS 2 helper for estimating the
dominant payload swing frequency from an AprilTag/TF position signal.

It subscribes to a `tf2_msgs/msg/TFMessage`, extracts one tag translation
component, and fits a zero-damping sinusoid:

```text
y(t) = c0 + c1*t + a*cos(omega*t) + b*sin(omega*t)
```

The output assumes:

```text
zeta = 0
T_shaper = pi / omega
period = 2*pi / omega
```

This is useful when the adaptive input shaper needs an online estimate of
`omega` but the damping estimate is too noisy to trust.

## Expected Input

Default topic:

```text
/apriltags/base_cam/rgb/tf
```

Message type:

```text
tf2_msgs/msg/TFMessage
```

The script expects one or more transforms like:

```text
header.frame_id: base_cam_rgb_camera_optical_frame
child_frame_id: tag36h11:6
transform.translation.x/y/z
```

By ROS convention, TF translation is in meters. The script converts to
millimeters with `--input-scale 1000`.

## Basic Use

```bash
source /opt/ros/humble/setup.bash

python3 zero_zeta_tf_system_id.py \
  --topic /apriltags/base_cam/rgb/tf \
  --child-frame-id tag36h11:6 \
  --axis x \
  --input-scale 1000 \
  --history-s 20 \
  --warmup-s 4 \
  --half-period-min-s 0.5 \
  --half-period-max-s 4.0 \
  --estimate-topic /payload/zero_zeta_id
```

Use `--axis x` for left/right image motion, `--axis y` for vertical image
motion, `--axis z` for depth motion, or `--axis xy` for planar magnitude.

If the TF message contains multiple tags, always set `--child-frame-id`; the
order of transforms in a `TFMessage` is not guaranteed.

## Output

The estimator publishes:

```text
/payload/zero_zeta_id
std_msgs/msg/Float64MultiArray
```

Fields:

```text
time,
omega_rad_s,
zeta,
shaper_T_s,
osc_period_s,
freq_hz,
amplitude_mm,
rmse_mm,
nrmse,
p2p_mm,
num_samples,
sample_rate_hz
```

`zeta` is always `0.0` because this method intentionally estimates only
frequency.

## Logging

To save estimates:

```bash
python3 zero_zeta_tf_system_id.py \
  --topic /apriltags/base_cam/rgb/tf \
  --child-frame-id tag36h11:6 \
  --axis x \
  --log-csv zero_zeta_estimates.csv
```

## Sampling-Rate Notes

At about `3.6 Hz`, the estimator can work only for slow payload swings. A
higher camera/TF rate is strongly preferred:

- `10 Hz` is a practical minimum.
- `20-30 Hz` is much better.

The script rejects candidate frequencies that are too fast for the observed
sample rate and prints a warning when there are too few samples per cycle.

## Useful ROS Checks

```bash
ros2 topic hz /apriltags/base_cam/rgb/tf
ros2 topic echo /apriltags/base_cam/rgb/tf --once
ros2 topic echo /payload/zero_zeta_id
```

