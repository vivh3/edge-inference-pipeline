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
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd /tmp && rm -rf demo-clone
tput cols; tput lines      # want at least 120 and 36; resize the window if not
asciinema rec ~/edge-inference-pipeline/results/demo/core.cast
```

The recording takes the size of the terminal it starts in. apt's asciinema on
JetPack is 2.1.0, which has no `--cols` or `--rows`, so size the window first.

`PYTEST_DISABLE_PLUGIN_AUTOLOAD` is there because `env.sh` sources ROS, and
ROS Humble ships a pytest plugin, `launch_testing`, written for an older hook
signature than pytest 9 accepts. With it on the path pytest stops before
collecting anything. The tests use no plugins, so turning autoload off costs
nothing, and the command in the shot stays a plain `pytest -q`.

`PS1` is reset because the recording captures the prompt, and the default one
carries the username and hostname. asciinema starts a new shell that brings
the default prompt back, so the first line typed in the recording is:

```bash
export PS1='$ ' && clear
```

`clear` hides the old prompt on screen but the cast still holds it, so the
committed `core.cast` was trimmed to start at the `clear` and end 4 s after
the table. Check any cast before committing it:

```bash
grep -c "$USER@" ~/edge-inference-pipeline/results/demo/core.cast    # want 0
```

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

**4.** When the table prints, wait 4 s on it, then stop with `Ctrl-D`. Wait
for `asciinema: recording finished`; until then everything typed still goes
into the cast.

Leave the keyboard alone while the simulation runs. Stray keys are recorded.

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

Open two terminals on the board. One will be recorded. The other is for
watching the pipeline while you move the camera, and is never recorded.

### Before recording: start the graph

**1.** Start five stacked panes and the four nodes. Run this outside tmux:

```bash
cd ~/edge-inference-pipeline
tput cols; tput lines      # want at least 160 and 48
R='cd ~/edge-inference-pipeline && source env.sh && export PS1="$ " && clear && cd ros2_ws'
tmux new-session -d -s demo
for i in 1 2 3 4; do tmux split-window -t demo; tmux select-layout -t demo even-vertical; done
tmux send-keys -t demo.0 "$R && python3 -m edge_perception.telemetry_node --ros-args -p out_dir:=$HOME/edge-inference-pipeline/results/demo" Enter
tmux send-keys -t demo.1 "$R && python3 -m edge_perception.capture_node" Enter
tmux send-keys -t demo.2 "$R && python3 -m edge_perception.inference_node --ros-args -p use_mock_engine:=false" Enter
tmux send-keys -t demo.3 "$R && python3 -m edge_perception.consumer_node --ros-args -p stale_after_s:=13.5 -p no_data_after_s:=20.0" Enter
tmux send-keys -t demo.4 "$R && ros2 topic list > /dev/null; clear" Enter
```

Top to bottom: telemetry (0), capture (1), inference (2), consumer (3), and a
spare (4) for the two commands typed during the recording. The panes are
stacked full width because the report lines are about 150 characters long.

`out_dir` on the telemetry node is not optional. Its default is
`results/sustained`, and a demo run would overwrite the committed Gate 2 data.

The stale limit is 13.5 s because of how old a record gets in normal
operation. It arrives 6.5 s old and the next one lands 6.45 s
later, so every record reaches about 12.9 s before it is replaced. Any limit
below that puts the consumer in `hold` for part of every cycle with the
pipeline healthy, and the scene-driven transition in shot 3 disappears under
it.

`ros2 topic list` in the spare pane starts the ROS command-line daemon. The
first `ros2` command in a session can run before the daemon has found the
topics, and then prints `does not appear to be published yet`. Running one
beforehand keeps that out of shot 4.

**2.** Watch the answers arrive, in the second terminal:

```bash
tail -f ~/edge-inference-pipeline/results/demo/results.csv | cut -d, -f2,16,17,20,21
```

One line per result: time, valid, failure kind, `path_status`,
`obstacle_location`. Nothing prints until the model has loaded.

**3.** Find three camera positions and remember them:

| position | reads as | what worked here |
|---|---|---|
| clear | `True,none,clear,none` | open floor, nothing near the frame edges |
| blocked | `True,none,blocked,<where>` | a view with an object at one side |
| unusable | `False,unusable_semantics` | camera face down on the desk |

Hold each view for three lines before judging it. The first line after a move
usually describes the old view. An object held close to the lens does not read
as an obstacle; the model called that `clear`.

**4.** Leave the camera on clear, wait for the consumer to report `proceed`,
and type the shot 4 command into the spare pane without running it:

```bash
tmux send-keys -t demo.4 "ros2 topic echo /perception --once --full-length"
tmux select-pane -t demo.4
```

Without `--full-length`, `ros2 topic echo` cuts the record off before the
`validation` block, which is the part the shot is for.

### Now record

Every answer lands 7 to 13 s after the camera moves: the inference in flight
has to finish, then the next one runs. Move the camera, keep still, and wait
for the line in the second terminal or the consumer pane before moving again.

**5.** Start the recording, in the first terminal:

```bash
asciinema rec -c "tmux attach -t demo" ~/edge-inference-pipeline/results/demo/graph.cast
```

Recording the attach keeps the shell prompt out of the cast, and the recording
ends when you detach.

**6. Shot 3.** Do not type. Wait for a new `records=` line in the telemetry
pane, which reports every 30 s. Then turn the camera to the blocked position,
wait for `proceed -> hold` in the consumer pane, turn it back, and wait for
`hold -> proceed`.

What the frame contains, and what to caption:

| pane | the number |
|---|---|
| capture | `camera negotiated` at 30 fps, the rate the sensor offers |
| inference | `policy_dropped` past 98%, `middleware_lost` reported **separately**, `health=healthy` |
| telemetry | `result_age p50` flat **while the drop rate climbs**: 6.2 s in the committed take, 6.46 s on the richer scene of the sustained run |
| consumer | `proceed` / `hold` following the scene |

Caption: 183x overload, both losses named, result age held near one service
time. That is the responsiveness claim, visible.

**7. Shot 4.** Lay the camera face down. Wait for two
`False,unusable_semantics` lines in the second terminal, then press `Enter` in
the recorded window. The spare pane prints the next record in full:

```
"semantic": {"path_status": "unknown", "obstacle_location": "unknown"},
"validation": {"ok": false, "failure": "unusable_semantics", ...}
```

and the consumer pane shows `proceed -> hold: semantics unusable`. The model
emitted something self-contradictory and the system published an explicit
unknown, named the cause, and kept the trusted `frame_id` and timestamps
beside it.

Caption: 10% of varied frames do this. **This is the shot most demos skip.**

**8. Shot 5.** Point the camera back at clear and wait for `-> proceed`. Then
type this in the spare pane, press `Enter`, and do not touch anything for
~20 s:

```bash
pkill -9 -f edge_perception.inference_node
```

The inference pane prints `Killed`, and the consumer pane prints, with nothing
arriving to cause either line:

```
proceed -> hold: last result 13.6s old, not a current view
hold -> no_data: last result 20.1s old, past the 20s limit
```

The ages land within a tick of the limits because the consumer decides at
10 Hz. The first line comes 1 to 7 s after the kill, depending on how old the
last record already was.

The consumer aged out its own last record. One that re-evaluated only on
arrival would still be reporting `proceed`.

Use the kill command. `Ctrl-C` in the inference pane interrupts it
mid-inference and ends in a traceback and `Aborted (core dumped)`, which looks
like a crash.

Caption that the limits are 13.5 s and 20 s for the shot; the defaults are
13 s and 30 s. Thirty seconds of dead air does not belong in a ninety-second
video, and neither does implying the defaults were recorded.

**9.** Stop by detaching. The recording ends with it:

```
Ctrl-B d
```

Then `tmux kill-session -t demo` stops the nodes.

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

Shots 1 and 4 are repeatable immediately, and shot 2 costs four minutes. Shot
5 kills the engine, so redoing it costs another ~100 s model load. For a
second take, rerun the inference line in pane 2, wait for the telemetry pane
to report again, and add `--overwrite` to `asciinema rec`.

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
