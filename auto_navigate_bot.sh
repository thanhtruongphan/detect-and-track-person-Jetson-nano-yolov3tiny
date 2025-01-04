#!/bin/bash

cd ~/linorobot2/docker || exit

# Docker compose build
sudo docker compose build

# Docker compose up - bring all required services
sudo docker compose up -d webtop bringup navigate debug rviz-nav slam
