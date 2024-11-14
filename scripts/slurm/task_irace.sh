#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=15
#SBATCH --mem=1000
#SBATCH --job-name=irace_job

enroot start -r -m experiment:/experiment automode bash -c "cd /experiment && irace"
