#!/bin/bash

# Bash strict mode, to catch problems and bugs in the shell
# script. See http://redsymbol.net/articles/unofficial-bash-strict-mode/.
set -euo pipefail

# No manual feedback for apt-get:
export DEBIAN_FRONTEND=noninteractive

# Update the package listing:
apt-get update

# Install security updates:
apt-get -y upgrade

apt-get update --fix-missing && \
apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
