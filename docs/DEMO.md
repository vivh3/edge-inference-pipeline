# Demo: what to record, and what each shot has to show

The video is not "look, it runs". It has one job:

> **A well-engineered system stays responsive and honest when an expensive
> learned component cannot keep pace with its sensor.**

Every shot below exists to carry part of that sentence. A shot that only shows
the thing working is cut.

Target: **90 seconds** after the edit, no narration required -- the terminal
output says it. The raw recordings run to about seven minutes, most of it shot
2 waiting for a queue to fill, which the edit speeds up.

---

## Install once

```bash
sudo apt install -y asciinema tmux
```

Record the terminal as text, not by filming a monitor: it stays legible at any
size, the file is kilobytes, it diffs, and a reader can replay it at their own
speed and copy a line out of it. All four false of a video of a screen.

---

# Recording 1 — shots 1 and 2 (about 4.5 min raw)

One pane, no hardware. It needs Python 3.10+ and pytest 7 or newer, nothing
else. On the Jetson that means the venv: apt's pytest is 6.2.5 and fails on
import.

**1.** Set up the shell and start the recording:

```bash
cd ~/edge-inference-pipeline && source env.sh
mkdir -p results/demo
export PS1='$ '
cd /tmp && rm -rf demo-clone
asciinema rec ~/edge-inference-pipeline/results/demo/core.cast --cols 120 --rows 36
```

`PS1` is reset because the recording captures the prompt, and the default one
carries the username and hostname. asciinema starts a new shell, so set it
again as the first line of the recording if the prompt comes back.

**2.** Shot 1 — type this and let it finish (about 12 s):

```bash
git clone --depth 1 https://github.com/vivh3/edge-inference-pipeline demo-clone
cd demo-clone && python3 -m pytest -q
```

Wait for `133 passed`. That is the shot: no accelerator, no ROS, no model
download. Do not scroll.

**3.** Shot 2 — type this and let it finish (about 4 min; speed it up in the
edit):

```bash
python3 tools/run_overload_sim.py --policies latest fifo_bounded \
    --duration 120 --latency 6.09 --sigma 0.02 --deadline 15
```

Two policies, 120 s each. The duration is not padding: an 8-deep queue takes
nine service times, 55 s, to fill, and until it has filled a bounded FIFO
looks like an unbounded one. A 30 s run prints 18 s for `fifo_bounded` and the
caption below would not match the screen.

**4.** When the table prints, wait 4 s on it, then stop:

```
Ctrl-D
```

**What to caption in the edit.** Not the drop rate — both policies drop ~99%.
The line is: `latest` answers in 6.1 s and `fifo_bounded` in 55 s while
publishing the same number of results. *Dropping is not what separates them;
which frames survive is.* Also caption `SIMULATED` — synthetic engine, real
policies. Quote the figures the recording shows; the README's 54.798 s is the
240 s run.

---

# Recording 2 — shots 3, 4 and 5 (about 2 min raw)

On the Jetson. The model takes ~100 s to load, so **warm the graph first and
record an attach**. Nothing in the recording waits for a load.

### Before recording: start the graph

**1.** Find two camera positions before starting: one the model reads as
`clear` (open floor) and one it reads as `blocked`. Start on the clear one.
Not a blank surface yet; that is shot 4.

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
cd ~/edge-inference-pipeline && source env.sh && export PS1='$ '
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
# pane 4 — consumer, with the no-data limit shortened so shot 5 fits the video
cd ros2_ws && python3 -m edge_perception.consumer_node --ros-args     -p stale_after_s:=13.5 -p no_data_after_s:=20.0
```

`out_dir` on the telemetry node is not optional. Its default is
`results/sustained`, and a demo run would overwrite the committed Gate 2 data.

The stale limit is 13.5 s, not something shorter, because of how old a record
gets in normal operation. It arrives 6.5 s old and the next one lands 6.45 s
later, so every record reaches about 12.9 s before it is replaced. Any limit
below that puts the consumer in `hold` for part of every cycle with the
pipeline healthy, and the scene-driven transition in shot 3 disappears under
it.

Leave **pane 5** empty. It is for the one command in shot 4.

**5.** Wait for pane 1 to print a `records=` line (~100 s for the model load,
then up to 30 s for the next report), and for pane 4 to print `-> proceed`.
Do not start recording before both.

**6.** Detach:

```
Ctrl-B d
```

### Now record

**7.** Start the recording and attach straight back:

```bash
asciinema rec ~/edge-inference-pipeline/results/demo/graph.cast --cols 160 --rows 48
tmux attach -t demo
```

**8. Shot 3 — about 35 s.** Do not type. The panes already hold report lines
from the warm-up; inference adds one every 10 s and telemetry one every 30 s,
so wait for a new telemetry line. Then move the camera to the blocked
position, wait for pane 4 to print `proceed -> hold` (up to two cycles, 13 s),
and move it back.

What the frame contains, and what to caption:

| pane | the number |
|---|---|
| capture | `camera negotiated` at 30 fps, the rate the sensor offers |
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

**10. Shot 5 — point the camera back at the clear position and wait for pane
4 to print `-> proceed`.** Shot 4 left the consumer in `hold`, and from there
only the second line below would print. Then go to pane 3 and `Ctrl-C` only
`inference_node`, watch pane 4, and do not touch anything for ~20 s.

It prints, with nothing arriving to cause either line:

```
proceed -> hold: last result 13.6s old, not a current view
hold -> no_data: last result 20.1s old, past the 20s limit
```

The ages land within a tick of the limits, 13.5 to 13.6 and 20.0 to 20.1,
because the consumer decides at 10 Hz. The first line comes 1 to 7 s after the
kill, depending on how old the last record already was.

The consumer aged out its own last record. One that re-evaluated only on
arrival would still be reporting `proceed`.

If no camera position reads `clear`, record it anyway: `hold -> no_data` alone
still shows a state change with nothing arriving.

Caption that the limits are 13.5 s and 20 s for the shot; the defaults are
13 s and 30 s. Thirty seconds of dead air does not belong in a ninety-second
video, and neither does implying the defaults were recorded.

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
cd ~/edge-inference-pipeline
git add results/demo/core.cast results/demo/graph.cast
git commit -m "Demo recordings"
```

Commit the `.cast` files. They are text, they diff, and anyone can replay them
at their own speed. `.gitignore` keeps everything else in `results/demo/` out,
which covers the telemetry CSV the demo run writes there and anything dropped
in by hand. The phone video from recording 3 shows a room and stays out of the
repository for the same reason the probe frames do.

For something embeddable in the README, `asciinema upload <file>.cast` gives a
link, or install [`agg`](https://github.com/asciinema/agg) separately -- it is
not part of the `asciinema` package -- and convert:

```bash
agg results/demo/graph.cast results/demo/graph.gif
```

A gif is for the video edit and the README, and is the one other file type
`.gitignore` lets through. The `.cast` is the artifact.

### If a shot goes wrong

Shots 1 and 4 are repeatable immediately, and shot 2 costs four minutes. Shot 5 kills the engine, so
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
