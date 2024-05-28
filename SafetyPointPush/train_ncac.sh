#!/bin/bash
#SBATCH --job-name=ncac # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --output=test_job_ncac%j.out # Standard output and error log

 
 
python ncac.py
echo "FinishedTraining"
