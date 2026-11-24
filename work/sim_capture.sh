#!/bin/bash
# Gazebo (separate ROS domain), both arms to the facing pose, 8 relative poses, two wrist frames each
source ~/garment_ws/install/setup.bash; export ROS_DOMAIN_ID=79
W=$HOME/foundationpose/work
if ps -eo args | grep -E "ros2_control_node.*__ns:=/(left|right)|realsense2_camera_node" | grep -v grep > /dev/null; then echo "REAL ROBOT RUNNING: not started"; exit 2; fi
ros2 launch garment_connector connector_sim.launch.py gui:=false camera:=lowres connector_color:=real > $W/sim_out/sim.log 2>&1 &
LP=$!
for i in $(seq 1 30); do timeout 5 ros2 topic list 2>/dev/null | grep -q "right_cartesian_compliance_controller/current_pose" && break; sleep 3; done
sleep 5
C=$(ros2 pkg prefix garment_connector)/share/garment_connector/config
timeout 60 ros2 run garment_connector go_to_joints --ros-args -p preset:=home -p duration:=5.0 -p max_wrist_deg:=360.0 2>&1 | grep -c reached
timeout 60 ros2 run garment_connector go_to_pose --ros-args -p preset_file:=$C/sim_poses.yaml -p preset:=home_facing_near -p duration:=4.0 2>&1 | grep -c reached
timeout 40 python3 $W/sim_grab.py $W/sim_inputs 00
i=1
for p in "0.004 0.0 0.0 0 0 0" "-0.008 0.005 0.0 0 0 0" "0.004 -0.008 0.005 5 0 0" "0.0 0.006 -0.01 -5 6 0" "0.006 0.0 0.008 0 -6 10" "-0.006 -0.004 -0.003 6 0 -10" "0.0 0.0 0.0 -6 -6 0" "0.003 0.003 0.004 4 4 6" "-0.004 0.002 -0.006 -4 3 -8" "0.002 -0.004 0.002 3 -5 5" "0.0 0.004 0.0 -3 2 -4"; do
  set -- $p
  timeout 40 ros2 run garment_connector go_to_pose --ros-args -p side:=right -p "xyz:=[$1, $2, $3]" -p "rpy_deg:=[$4.0, $5.0, $6.0]" -p duration:=2.0 > /dev/null 2>&1
  sleep 1; timeout 40 python3 $W/sim_grab.py $W/sim_inputs $(printf %02d $i) | cut -c1-90; i=$((i+1))
done
kill -INT $LP; sleep 6
for p in $(pgrep -u $USER -f "ign gazebo.*garment_connector"); do kill -INT $p; done
sleep 2; echo "frames: $(ls $W/sim_inputs/*.npz | wc -l)"; echo CAPTURE_DONE
