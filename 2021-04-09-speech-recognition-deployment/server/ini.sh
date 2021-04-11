#!/bin/bash

sudo apt-get update

# install docker
sudo apt install docker.io
sudo apt install apt-utils

# start dokcer service
sudo systemctl start dokcer
sudo systemctl enable docker

# install docker compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# build and run docker containers
cd ~/server 
sudo docker-compose build
sudo docker-compose up
