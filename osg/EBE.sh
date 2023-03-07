#!/bin/bash
# EBE.sh: a single event
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job running as user: "; /usr/bin/id
printf "Job is running in directory: "; /bin/pwd
echo
echo "Working hard..."
# Start a new bash terminal session to refresh the path
bash
# Activate the python virtual environment
source jma/bin/activate
# Change directory to the project root
cd /jm-analysis || exit
# Pull any changes from the git
git pull
# Run the script
python3 EBE.py
echo "Science complete!"