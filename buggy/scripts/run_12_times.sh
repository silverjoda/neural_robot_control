#!/bin/bash
echo "Sleeping 15 secs"
sleep 15
for ((i = 0 ; i <= 15 ; i++)); do
  python3 /home/pi/SW/neural_robot_control/buggy/src/barinov_buggy_control.py
done
