#!/bin/bash
# EBE.sh: a single event
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job running as user: "; /usr/bin/id
printf "Job is running in directory: "; /bin/pwd  # Should be /srv by default
echo
echo "Working hard..."
# Activate the python virtual environment
export PATH=/usr/bin:$PATH
source /usr/jma/bin/activate
# Run the script
python3 /usr/jm-analysis/EBE.py
echo "Event complete!"