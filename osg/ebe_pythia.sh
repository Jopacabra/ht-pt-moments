#!/bin/bash
# EBE.sh: a single event
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job running as user: "; /usr/bin/id
printf "Job is running in directory: "; /bin/pwd  # Should be /srv by default
echo

# Export osu-hydro and trento binary location to path
export PATH=/usr/bin:$PATH

# Activate the conda environment
echo "Activating conda environment..."
conda init
source ~/.bash_profile
source ~/.bashrc
conda activate /usr/conda/jma

# Run the script
echo "Running event..."
python3 /usr/jm-analysis/ebe_pythia.py
echo "Event complete!"

# Transfer relevant output files to /srv
echo "Moving results parquet to /srv..."
mv /srv/results/*/*.pickle /srv
mv /srv/results/*/*.dat /srv

echo "Job complete! Have a great day! :)"
