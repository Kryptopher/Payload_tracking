# Standalone Wall Tracker

This tracker is separate from the ROS payload tracker. It uses a fixed OAK-D Pro
PoE camera to track a rigid AprilTag target and streams an annotated camera view
plus a top-down X/Y plot.

## Tag Layout

The cube target is a `100 mm x 100 mm x 100 mm` box with 40 mm AprilTags. Use
3 tags on each vertical side and no tags on the top or bottom:

```text
front:  ID 0,  ID 1,  ID 2
right:  ID 3,  ID 4,  ID 5
back:   ID 6,  ID 7,  ID 8
left:   ID 9,  ID 10, ID 11
```

The cube layout in `tag_layout_box_100mm.json` assumes:

- Tag family: `tagStandard41h12`
- Tag size: `0.040 m`, measured across the outer black square
- Box size: `0.100 m` per side
- Target center: `(0.0, 0.0, 0.0) m`, the geometric center of the cube
- Coordinate frame: `+X` right face, `+Y` up, `+Z` front face

Each side uses a compact triangle:

```text
          ID 2


    ID 0        ID 1
```

The old planar wall layout remains available in `tag_layout.json` and assumes:

- Tag family: `tagStandard41h12`
- Tag size: `0.10 m`, measured across the outer black square
- Target center: `(0.0, 0.0, 0.0) m` in the layout frame
- ID 0 at `(-0.20, -0.12, 0.0) m`
- ID 1 at `( 0.20, -0.12, 0.0) m`
- ID 2 at `( 0.00,  0.18, 0.0) m`

Measure your installed tag centers and update the layout file if the spacing
differs. For cube layouts, each tag defines `normal_m` and `up_m` so the tracker
knows which cube face the tag belongs to. The `CENTER` marker and X/Y top-down
plot represent `target_center_m`, not the currently visible face.

## Run

Install dependencies in your Python environment:

```bash
python3 -m pip install -r requirements.txt
```

Start the cube tracker:

```bash
./run_wall_tracker.sh
```

Then open:

```text
http://host:8090/
```

## Camera IP

The tracker connects to the OAK camera at `192.168.0.153` by default. To use a
different camera, set `OAK_IP` when launching:

```bash
OAK_IP=<CAMERA_IP> ./run_wall_tracker.sh
```

You can also pass the IP directly to the Python programs:

```bash
python3 wall_tracker.py --ip <CAMERA_IP> --layout tag_layout_box_100mm.json
python3 calibrate_cube_layout.py --ip <CAMERA_IP>
```

The web stream address is separate from the camera IP. Open
`http://host:8090/`, replacing `host` with the computer running the tracker. To
change the web stream port, set `STREAM_PORT` or pass `--stream-port`.

Useful overrides:

```bash
OAK_IP=<CAMERA_IP> STREAM_PORT=8090 ./run_wall_tracker.sh

./run_wall_tracker.sh -- \
  --layout tag_layout_box_100mm.json

./run_wall_tracker.sh -- \
  --detect-width 800 \
  --detect-height 500 \
  --detect-fps 90

./run_wall_tracker.sh -- --high-fps

./run_wall_tracker.sh -- \
  --display-fps 8 \
  --jpeg-quality 60 \
  --stream-from-detect-only

./run_wall_tracker.sh -- \
  --tag-family tagStandard41h12 \
  --tag-size 0.10

./run_wall_tracker.sh -- \
  --quad-decimate 1.5 \
  --decision-margin-min 8 \
  --hold-sec 0.35

./run_wall_tracker.sh -- --topdown-mode relative

./run_wall_tracker.sh -- --no-depth

./run_wall_tracker.sh -- --camera-socket CAM_A --no-depth

./run_wall_tracker.sh -- \
  --depth-roi-px 25 \
  --depth-min-m 0.20 \
  --depth-max-m 8.0

./run_wall_tracker.sh -- \
  --graph-deadband-m 0.012 \
  --max-depth-age-sec 0.12

./run_wall_tracker.sh -- --no-single-tag-fallback
```

## Axes And Pose

The camera frame follows the OpenCV/OAK optical convention:

- `X`: image right
- `Y`: image down/up in the camera optical pose convention
- `Z`: depth away from the camera

The stream reports both absolute position and relative displacement:

- `absolute`: `X`, `Y(range)`, `Z` in meters; `Y(range)` is the OAK stereo range
- `relative`: `dX`, `dY`, `dZ` from the locked origin
- Default top-down plot: stabilized absolute `X_abs` versus `Y_abs`, so graph
  `Y_abs` follows the displayed stereo depth/range semantics without plotting
  tiny per-frame jitter
- Relative top-down mode: `--topdown-mode relative`, labeled `dX/dY`
- `yaw`: in-plane rotation in the camera view
- `roll` and `pitch`: out-of-plane tilt
- `omega`: in-plane yaw rate, shown in both `rad/s` and `deg/s`
- `depth`: `stereo` when the OAK depth map is valid, otherwise `pnp fallback`
- `pose`: `multi-tag` when the configured 3-tag layout fits, or
  `single-tag ID...` when the rigid layout is inconsistent and the tracker is
  recovering from one visible tag
- `CENTER`: yellow marker for the tracked translation point; this uses
  `target_center_m` during multi-tag tracking and the visible tag centroid during
  single-tag recovery
- `center`: `layout` when the configured `target_center_m` is being projected
  from the multi-tag pose, or `visible-tags` when the tracker falls back to the
  detected tag centroid during single-tag recovery
- `origin`: shown in relative mode; `warming` while the tracker waits for a
  stable stereo-backed zero point, then `locked`

Translation and rotation come from the same rigid PnP solve each frame, so
combined motion is handled as one pose estimate rather than separate heuristics.
Stereo depth is used to improve the displayed forward `Y` translation; PnP is
still the fallback when stereo pixels are invalid or depth is disabled.

For the cube target, all 12 tags share one rigid cube-centered model. When the
visible face changes, the tracker solves for the same geometric center instead
of resetting translation to the new face. A face handoff during in-place
rotation should mostly change roll, pitch, or yaw while keeping the top-down
center stable.

## Sampling Rate

The fast path is:

- Grayscale OAK mono detection stream
- AprilTag detection at `640x400` by default
- Stream composition and JPEG encoding throttled to `--display-fps`
- Display stream at lower FPS than detection, or detection-frame reuse with
  `--stream-from-detect-only`
- Optional OAK stereo depth stream sampled around the target center
- Pose solve only for the configured tag IDs
- No ROS publishers or dashboard integration
- Planar PnP candidate continuity so fast rotations prefer the closest plausible pose
- Short `HOLD` display during brief dropouts so the top-down view does not disappear

Use `--high-fps` first when tracking rate matters more than stream smoothness.
It requests a faster detection stream, lowers web stream FPS/quality, and avoids
requesting a separate camera display output. The stream still shows the camera
view and top-down graph, just at a lower update rate.

If detection is unstable at range, increase detection resolution before lowering
`--quad-decimate` further. If CPU is overloaded, lower `--display-fps`, raise
`--quad-decimate` back toward `2.0`, or add `--no-stream` for pose-only terminal
diagnostics. If stereo depth is noisy, adjust `--depth-roi-px`,
`--depth-min-m`, and `--depth-max-m`; use `--no-depth` to return to PnP-only
translation.

For FPS tuning, compare these tiers and watch the console `det=... stream=...`
readout:

```bash
# Tracking-priority stream.
./run_wall_tracker.sh -- --high-fps

# Same idea, but with manual stream settings.
./run_wall_tracker.sh -- --display-fps 8 --jpeg-quality 60 --stream-from-detect-only

# Find the detector-only ceiling without MJPEG work.
./run_wall_tracker.sh -- --no-stream

# Isolate stereo-depth cost if detection FPS is still low.
./run_wall_tracker.sh -- --high-fps --no-depth
```

If the top-down dot slowly wanders while the target is held still, increase
`--graph-deadband-m` slightly, for example to `0.012` or `0.015`. If the dot
lags too much during real motion, lower it toward the default.

The MJPEG view falls back to the detection frame if the separate display stream
stalls. If the video pane is still blank or white, try `--no-depth` first, then
try `--camera-socket CAM_A --no-depth` to use the center camera for AprilTag
detection.

If detections are visible but `pose=single-tag ID...` appears often, the tracker
is protecting you from a bad rigid-board solve. Measure the physical tag center
spacing and update `tag_layout.json` so IDs 0, 1, and 2 match the board exactly.
If the top-down point starts far from zero, check whether `origin=warming` has
changed to `origin=locked`; startup samples are intentionally ignored until the
stereo-backed center is stable.

## Validation

1. Start the tracker and confirm all 3 IDs appear in the overlay.
2. Hold the target still; reprojection error should stay low and pose should be stable.
3. Translate only; the top-down X/Y point should move without large angle changes.
4. Rotate in-plane; `yaw` should change most.
5. Tilt out-of-plane; `roll` or `pitch` should change most.
6. Move and rotate at the same time; the pose should remain `GOOD` or `WARN`.
7. Confirm the overlay shows translation, rotation, omega, and depth source.

Quality labels:

- `GOOD`: all 3 tags visible and reprojection error is within the gate.
- `WARN`: 2 tags visible; pose is accepted but less redundant.
- `HOLD`: briefly displaying the last accepted pose while detections recover.
- `REACQ`: fewer than 2 tags visible; detections are shown but pose is held.
- `BAD`: PnP failed or reprojection error exceeded the gate.
