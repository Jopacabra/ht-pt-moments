#!/bin/bash
# EBE.sh: a single event
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job running as user: "; /usr/bin/id
printf "Job is running in directory: "; /bin/pwd  # Should be /srv by default
echo

# Export osu-hydro and trento binary location to path
export PATH=/usr/bin:$PATH

# Activate the python virtual environment
echo "Activating python virtual environment..."
source /usr/jma/bin/activate

# Run the script
echo "Running event..."
python3 /usr/jm-analysis/EBE.py
echo "Event complete!"

# Transfer relevant output files to /srv
echo "Moving results pkl to /srv..."
mv /usr/jm-analysis/results/*/*.pkl /srv

echo "Job complete! Have a great day! :)"
