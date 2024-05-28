#!/bin/bash
#SBATCH --job-name=cac # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=test_job_cac%j.out # Standard output and error log

 
 
python cac.py
echo "FinishedTraining"
