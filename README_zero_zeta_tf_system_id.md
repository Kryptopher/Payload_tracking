# Zero-Zeta TF System ID

`zero_zeta_tf_system_id.py` is a standalone ROS 2 helper for estimating payload
swing parameters from an AprilTag/TF position signal.

It subscribes to a `tf2_msgs/msg/TFMessage`, extracts one tag translation
component, and fits one of two models.

Zero-zeta mode fits:

```text
y(t) = c0 + c1*t + a*cos(omega*t) + b*sin(omega*t)
```

Damped-grid mode fits:

```text
y(t) = c0 + c1*t + exp(-sigma*t)*
       (a*cos(omega_d*t) + b*sin(omega_d*t))
```

For damped-grid mode:

```text
omega_n = sqrt(omega_d^2 + sigma^2)
zeta = sigma / omega_n
T_shaper = pi / omega_d
```

For zero-zeta mode:

```text
zeta = 0
T_shaper = pi / omega
period = 2*pi / omega
```

Zero-zeta mode is useful when the adaptive input shaper needs an online
estimate of `omega` but the damping estimate is too noisy to trust. Damped-grid
mode is available when an approximate `zeta` estimate is useful.

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

The transform can be camera-relative if the camera is fixed. A fixed camera
frame is still a fixed coordinate system, so the swing frequency is unchanged.
Use the translation axis with the clearest oscillation.

## Basic Use

```bash
source /opt/ros/jazzy/setup.bash

python3 zero_zeta_tf_system_id.py \
  --topic /apriltags/base_cam/rgb/tf \
  --child-frame-id tag36h11:6 \
  --axis x \
  --input-scale 1000 \
  --fit-mode zero-zeta \
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

To also estimate damping ratio:

```bash
python3 zero_zeta_tf_system_id.py \
  --topic /apriltags/base_cam/rgb/tf \
  --child-frame-id tag36h11:6 \
  --axis x \
  --input-scale 1000 \
  --fit-mode damped-grid \
  --zeta-min 0.0 \
  --zeta-max 0.2 \
  --zeta-grid-count 25
```

This is a bounded grid search, not nonlinear optimization. It is intentionally
conservative and rejects poorly conditioned fits.

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
sample_rate_hz,
damped_omega_rad_s,
decay_rate_s_inv,
condition,
fit_method_code
```

`fit_method_code` is `0` for zero-zeta and `1` for damped-grid.

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
