# Demo: what to record, and what each shot has to show

The video is not "look, it runs". It has one job:

> **A well-engineered system stays responsive and honest when an expensive
> learned component cannot keep pace with its sensor.**

Every shot below exists to carry part of that sentence. A shot that only shows
the thing working is cut.

Target: **90 seconds**, no narration required -- the terminal output says it.

---

## One-time setup

```bash
sudo apt install -y asciinema tmux
```

Record the terminal as text rather than filming a monitor: it stays legible at
any size, the file is kilobytes, and a reader can copy a line out of it.

```bash
asciinema rec results/demo/run.cast --cols 120 --rows 40
# ... do the shots ...
# exit, or Ctrl-D, to stop
```

Four panes in one window, so one recording captures the whole graph:

```bash
tmux new -s demo
# Ctrl-B %   split vertically       Ctrl-B "   split horizontally
# Ctrl-B o   move between panes     Ctrl-B z   zoom one pane full screen
```

Each pane starts with `source env.sh`. Pane order: telemetry, capture,
inference, consumer.

Separately, a phone video of the camera and whatever it is pointed at. Thirty
seconds is plenty, and it is the only shot that needs a camera pointed at the
rig rather than at a screen.

---

## The shots

### 1. A stranger can run it — 10 s

```bash
git clone https://github.com/vivh3/edge-inference-pipeline && cd edge-inference-pipeline
python3 -m pytest -q
```

**Shows:** 133 tests passing with no accelerator, no ROS, no model download.
The core is stdlib-only, so the claim "reproducible from a fresh clone" is
demonstrated rather than asserted.

### 2. The overload, and why the policy is the point — 25 s

```bash
python3 tools/run_overload_sim.py --duration 30 --latency 6.09 --sigma 0.02
```

**Shows:** the three policies side by side at the project's real service time.
Zoom the table. The number to linger on is not the drop rate — all three drop
~99% — it is that `latest` answers in 6.1 s and `fifo_bounded` in 54.8 s while
publishing the same count. **Dropping is not what separates them; which frames
survive is.**

Say on screen: `SIMULATED` — synthetic engine, real policies.

### 3. The real graph under overload — 25 s

All four panes live, camera pointed at a scene.

**Shows, in one frame:**

- `capture_node` at 30 fps
- `inference_node`: `policy_dropped` climbing past 98%, `middleware_lost`
  reported *separately*, `health=healthy`
- `telemetry_node`: `result_age p50` flat near 6.46 s while the drop rate
  climbs — the responsiveness claim, visible
- `consumer_node`: `proceed`/`hold` transitions as the scene changes

The overload is 183x. Both losses are named. Nothing is hidden behind one
number.

### 4. Honest about a bad output — 15 s

Point the camera somewhere the model fails — a blank surface works, which is
how the face-down run was diagnosed.

```bash
ros2 topic echo /perception --once
```

**Shows:** `"path_status": "unknown"`, `"obstacle_location": "unknown"`, and a
`validation` block naming `unusable_semantics`. The model emitted something
self-contradictory and the system published an explicit unknown instead of
passing it on. 10% of varied frames do this.

This is the honesty half of the thesis, and it is the shot most demos skip.

### 5. Stale data is not current data — 15 s

With the graph running, `Ctrl-C` only `inference_node` and watch
`consumer_node`.

```bash
# in the consumer pane, started with shortened limits so the shot fits
python3 -m edge_perception.consumer_node --ros-args \
    -p stale_after_s:=8.0 -p no_data_after_s:=15.0
```

**Shows:** `proceed → hold: last result 8.4s old, not a current view`, then
`hold → no_data`. Nothing arrived to cause either transition — the consumer
aged out its own last record. A consumer that re-evaluated only on arrival
would still be reporting `proceed`.

State the shortened limits on screen. The defaults are 13 s and 30 s; 15 s of
dead air does not belong in a 90-second video, but neither does pretending the
defaults are what was recorded.

---

## After recording

```bash
# a gif for the README, or keep the .cast and link to asciinema.org
agg results/demo/run.cast results/demo/run.gif
```

Commit the `.cast`. It is text, it diffs, and anyone can replay it at their own
speed — all three false of a gif.

---

## What not to claim

- **Not real-time.** There is no deadline guarantee anywhere in this system.
  The word for 6.46 s is "responsive under overload", and only because the
  admission policy keeps result age near one service time.
- **Not safe.** `consumer_node` actuates nothing and `proceed` is not a safety
  function. Say so if the video shows the advisory at all.
- **Not a benchmark.** The latency figures describe this board at 15W with this
  model and this frozen generation policy. They are not a claim about the
  Jetson, about SmolVLM, or about VLMs on edge devices.
- **Not "I built a perception system".** What was built is overload semantics,
  explicit failure handling, and one root-caused bottleneck. The model is a
  dependency that was chosen and measured, not an achievement.
