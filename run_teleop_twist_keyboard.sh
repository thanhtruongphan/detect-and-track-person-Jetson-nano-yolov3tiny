#!/bin/bash

# Start the ROS Humble container
docker start ros-humble-container

# Attach to the container
docker exec -it ros-humble-container /bin/bash <<EOF
source ~/.bashrc
ros2 run teleop_twist_keyboard teleop_twist_keyboard
EOF
