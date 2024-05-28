#!/bin/bash
#SBATCH --job-name=dqn # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=test_job_dqn%j.out # Standard output and error log

 
 
python DQN.py
echo "FinishedTraining"
