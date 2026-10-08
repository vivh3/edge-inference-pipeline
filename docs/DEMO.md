# Demo: what to record, and what each shot has to show

The video is not "look, it runs". It has one job:

> **A well-engineered system stays responsive and honest when an expensive
> learned component cannot keep pace with its sensor.**

Every shot below exists to carry part of that sentence. A shot that only shows
the thing working is cut.

Target: **90 seconds**, no narration required -- the terminal output says it.

---

## Install once

```bash
sudo apt install -y asciinema tmux
```

Record the terminal as text, not by filming a monitor: it stays legible at any
size, the file is kilobytes, it diffs, and a reader can replay it at their own
speed and copy a line out of it. All four false of a video of a screen.

---

# Recording 1 — shots 1 and 2 (about 40 s)

One pane, no hardware. Do this anywhere, including not on the Jetson.

**1.** Start the recording:

```bash
mkdir -p ~/edge-inference-pipeline/results/demo
cd /tmp && asciinema rec ~/edge-inference-pipeline/results/demo/core.cast --cols 120 --rows 36
```

**2.** Shot 1 — type this and let it finish (about 12 s):

```bash
git clone --depth 1 https://github.com/vivh3/edge-inference-pipeline demo-clone
cd demo-clone && python3 -m pytest -q
```

Wait for `133 passed`. That is the shot: no accelerator, no ROS, no model
download. Do not scroll.

**3.** Shot 2 — type this and let it finish (about 32 s):

```bash
python3 tools/run_overload_sim.py --duration 30 --latency 6.09 --sigma 0.02
```

**4.** When the table prints, wait 4 s on it, then stop:

```
Ctrl-D
```

**What to caption in the edit.** Not the drop rate — all three policies drop
~99%. The line is: `latest` answers in 6.1 s and `fifo_bounded` in 54.8 s
while publishing the same number of results. *Dropping is not what separates
them; which frames survive is.* Also caption `SIMULATED` — synthetic engine,
real policies.

---

# Recording 2 — shots 3, 4 and 5 (about 60 s)

On the Jetson. The model takes ~100 s to load, so **warm the graph first and
record an attach**. Nothing in the recording waits for a load.

### Before recording: start the graph

**1.** Point the camera at a scene with structure — a doorway, a corridor,
objects on the floor. Not a blank surface yet; that is shot 4.

**2.** Build the five-pane layout:

```bash
tmux new -s demo
```

Then, inside tmux:

| keys | does |
|---|---|
| `Ctrl-B` `"` | split the current pane top/bottom |
| `Ctrl-B` `%` | split the current pane left/right |
| `Ctrl-B` `o` | move to the next pane |
| `Ctrl-B` `z` | zoom the current pane full screen (again to unzoom) |

Press `Ctrl-B "` then `Ctrl-B %`, then `Ctrl-B o` and `Ctrl-B %` again until
you have five panes. Exact arrangement does not matter; all five visible does.

**3.** In each pane, first:

```bash
cd ~/edge-inference-pipeline && source env.sh
```

**4.** Start the four nodes, one per pane, in this order:

```bash
# pane 1 — telemetry
cd ros2_ws && python3 -m edge_perception.telemetry_node --ros-args     -p out_dir:=$HOME/edge-inference-pipeline/results/demo
```
```bash
# pane 2 — capture
cd ros2_ws && python3 -m edge_perception.capture_node
```
```bash
# pane 3 — inference
cd ros2_ws && python3 -m edge_perception.inference_node --ros-args -p use_mock_engine:=false
```
```bash
# pane 4 — consumer, with shortened limits so shot 5 fits the video
cd ros2_ws && python3 -m edge_perception.consumer_node --ros-args     -p stale_after_s:=8.0 -p no_data_after_s:=15.0
```

Leave **pane 5** empty. It is for the one command in shot 4.

**5.** Wait for pane 1 to print its first `records=1` line (~100 s). Do not
start recording before this.

**6.** Detach:

```
Ctrl-B d
```

### Now record

**7.** Start the recording and attach straight back:

```bash
asciinema rec results/demo/graph.cast --cols 160 --rows 48
tmux attach -t demo
```

**8. Shot 3 — hold still for 25 s.** Do not type. Let two or three report
lines land in each pane. Move something in front of the camera once, so the
consumer pane shows a `proceed`/`hold` transition.

What the frame contains, and what to caption:

| pane | the number |
|---|---|
| capture | 30 fps, the sensor rate |
| inference | `policy_dropped` past 98%, `middleware_lost` reported **separately**, `health=healthy` |
| telemetry | `result_age p50` flat near 6.46 s **while the drop rate climbs** |
| consumer | `proceed` / `hold` following the scene |

Caption: 183x overload, both losses named, result age held near one service
time. That is the responsiveness claim, visible.

**9. Shot 4 — turn the camera to face a blank surface** (a desk, a wall).
Wait ~15 s, which is two inference cycles, then in pane 5:

```bash
ros2 topic echo /perception --once
```

Zoom it with `Ctrl-B z` while it is on screen, then `Ctrl-B z` again.

Look for `"path_status": "unknown"`, `"obstacle_location": "unknown"`, and a
`validation` block naming `unusable_semantics`. The model emitted something
self-contradictory and the system published an explicit unknown rather than
passing it on.

Caption: 10% of varied frames do this. **This is the shot most demos skip.**

**10. Shot 5 — go to pane 3 and `Ctrl-C` only `inference_node`.** Then watch
pane 4 and do not touch anything for ~20 s.

It prints, with nothing arriving to cause either line:

```
proceed -> hold: last result 8.4s old, not a current view
hold -> no_data: last result 15.2s old, past the 15s limit
```

The consumer aged out its own last record. One that re-evaluated only on
arrival would still be reporting `proceed`.

Caption that the limits are shortened to 8 s and 15 s for the shot; the
defaults are 13 s and 30 s. Fifteen seconds of dead air does not belong in a
ninety-second video, and neither does implying the defaults were recorded.

**11.** Stop:

```
Ctrl-B d
Ctrl-D
```

---

# Recording 3 — the rig (30 s, phone)

The camera, the Jetson, and whatever the camera is pointed at. The only shot
that points at hardware rather than a screen. Shoot it last, when you know
which scene shots 3 and 4 used, so the cut matches.

---

## Afterwards

```bash
git add results/demo && git commit -m "Demo recordings"
```

Commit the `.cast` files. They are text, they diff, and anyone can replay them
at their own speed.

For something embeddable in the README, `asciinema upload <file>.cast` gives a
link, or install [`agg`](https://github.com/asciinema/agg) separately -- it is
not part of the `asciinema` package -- and convert:

```bash
agg results/demo/graph.cast results/demo/graph.gif
```

A gif is for the video edit and the README. The `.cast` is the artifact.

### If a shot goes wrong

Shots 1, 2 and 4 are repeatable immediately. Shot 5 kills the engine, so
redoing it costs another ~100 s model load — do it last, and if you need a
second take, restart pane 3 and wait for pane 1 to report again before
recording.

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
