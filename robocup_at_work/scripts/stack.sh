#!/bin/bash
# Start or stop the whole simulation stack.
#
#   scripts/stack.sh up
#   scripts/stack.sh mission     # open gazebo with everything, run the mission
#   scripts/stack.sh down
#
# Readiness uses real service calls (ros2 cli lists can be stale).
# Teardown skips its own parent processes so it doesn't kill the caller.

source /opt/ros/humble/setup.bash
# workspace root, three levels up from this script
WS="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$WS/install/setup.bash"

LOGS="${TMPDIR:-/tmp}/robocup_stack"
mkdir -p "$LOGS"

HERE="$(cd "$(dirname "$0")" && pwd)"
wait_for() { python3 "$HERE/wait_for.py" "$@"; }

down() {
  mine=""
  p=$$
  while [ "$p" -gt 1 ]; do
    mine="$mine $p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -z "$p" ] && break
  done
  # every node the stack starts must be listed, leftover duplicates
  # break discovery
  for pat in gzserver gzclient robot_gazebo.launch moveit.launch \
             perception.launch navigation.launch spawn_entity \
             robot_state_publisher controller_manager ros2_control_node \
             spawner object_picker object_detector move_group mission \
             workstation_identifier apriltag_node nav2 bt_navigator \
             planner_server controller_server behavior_server amcl \
             map_server lifecycle_manager rviz2; do
    for pid in $(pgrep -f "$pat"); do
      case " $mine " in *" $pid "*) continue;; esac
      kill "$pid" 2>/dev/null
    done
  done
  sleep 3
  for pid in $(pgrep -f gzserver); do
    case " $mine " in *" $pid "*) continue;; esac
    kill -9 "$pid" 2>/dev/null
  done
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  ros2 daemon stop >/dev/null 2>&1
  sleep 1
  echo "down"
}

up() {
  rm -f /dev/shm/fastrtps_* 2>/dev/null
  ros2 daemon stop >/dev/null 2>&1

  nohup ros2 launch robocup_at_work robot_gazebo.launch.py \
        gui:="${GUI:-false}" > "$LOGS/gazebo.log" 2>&1 &
  wait_for --timeout 120 \
      --action /panda_arm_controller/follow_joint_trajectory \
      --action /panda_hand_controller/follow_joint_trajectory || {
    echo "controllers never came up, see $LOGS/gazebo.log"; return 1; }
  echo "gazebo and controllers up"

  nohup ros2 launch robocup_at_work perception.launch.py \
        > "$LOGS/perception.log" 2>&1 &
  wait_for --timeout 90 --topic /objects/poses || {
    echo "perception never came up, see $LOGS/perception.log"; return 1; }
  echo "perception up"

  nohup ros2 launch robocup_at_work moveit.launch.py \
        > "$LOGS/moveit.log" 2>&1 &
  # wait for move_group too, the picker services come up before it
  wait_for --timeout 120 --graph-service /compute_cartesian_path \
                         --graph-service /apply_planning_scene || {
    echo "move_group never came up, see $LOGS/moveit.log"; return 1; }
  wait_for --timeout 120 --service /pick/execute --service /pick/place || {
    echo "picker never appeared, see $LOGS/moveit.log"; return 1; }
  echo "picker up"

  # navigation is off by default, NAV=1 to start it (needed for mission)
  if [ "${NAV:-0}" = "1" ]; then
    nohup ros2 launch robocup_at_work navigation.launch.py \
          > "$LOGS/navigation.log" 2>&1 &
    # wait for the costmaps as well, the action server comes up first
    wait_for --timeout 180 \
        --graph-service /navigate_to_pose/_action/get_result \
        --graph-service /global_costmap/clear_entirely_global_costmap \
        --graph-service /local_costmap/clear_entirely_local_costmap || {
      echo "navigation never came up, see $LOGS/navigation.log"; return 1; }
    sleep 5
    echo "navigation up"
  fi
  echo "stack ready, logs in $LOGS"
}

mission() {
  NAV=1
  GUI="${GUI:-true}"
  up || return 1
  echo "running the mission, takes about 5 minutes"
  ros2 service call /mission/run std_srvs/srv/Trigger
}

case "${1:-up}" in
  up) up;;
  mission) mission;;
  down) down;;
  *) echo "usage: $0 [up|mission|down]"; exit 2;;
esac
