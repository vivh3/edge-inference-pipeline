# Per-shell setup for this repo. SOURCE it, do not run it:
#
#     source env.sh
#
# Running it would activate the venv in a subshell that exits immediately,
# leaving the calling shell exactly as it was -- which looks like the script
# doing nothing. There is a check for that below.
#
# Order matters. The venv goes first so its interpreter wins, then ROS, then
# the workspace overlay. ROS's setup prepends its own Python paths, and the
# venv must already be active when it does.

# Sourced, or run? $0 is the shell's name when sourced, the script's when run.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "source this file, do not run it:  source env.sh" >&2
    exit 1
fi

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Safe to source into a shell that already has this venv active, or a different
# one. `activate` runs `deactivate nondestructive` before it records the PATH to
# restore, so re-sourcing undoes the previous activation rather than stacking on
# it. Verified: PATH holds one copy of the venv's bin in either case, and
# `deactivate` afterwards leaves none. No guard here on purpose -- the obvious
# one to add would guard against a problem that does not exist.
if [ -f "$_repo/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_repo/.venv/bin/activate"
else
    echo "no .venv at $_repo (see docs/SETUP-jetson.md)" >&2
fi

# Skipped when already sourced to save repeating the work, not to avoid harm:
# colcon's generated scripts prepend with a uniqueness check, so a second source
# does not duplicate AMENT_PREFIX_PATH either.
if [ -f /opt/ros/humble/setup.bash ] && [ "${ROS_DISTRO:-}" != "humble" ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

_overlay="$_repo/ros2_ws/install"
if [ -f "$_overlay/setup.bash" ]; then
    case ":${AMENT_PREFIX_PATH:-}:" in
        *":$_overlay/edge_perception:"*) ;;
        *)
            # shellcheck disable=SC1091
            source "$_overlay/setup.bash"
            ;;
    esac
elif [ -n "${ROS_DISTRO:-}" ]; then
    echo "ros2_ws not built yet: colcon build --symlink-install" >&2
fi

# The nodes import `inference` and `telemetry` from the repo root, and they are
# run as `python3 -m edge_perception.<node>` from ros2_ws, so the root has to be
# on the path explicitly rather than by happening to be the directory.
case ":${PYTHONPATH:-}:" in
    *":$_repo:"*) ;;
    *) export PYTHONPATH="$_repo${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

printf 'repo      %s\n' "$_repo"
printf 'python    %s\n' "$(command -v python3)"
printf 'ros       %s\n' "${ROS_DISTRO:-not sourced}"
printf 'workspace %s\n' \
    "$([ -f "$_overlay/setup.bash" ] && echo overlaid || echo "not built")"
unset _repo _overlay
