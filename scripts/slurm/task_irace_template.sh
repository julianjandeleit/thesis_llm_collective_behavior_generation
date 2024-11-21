#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=[RUNTIME]
#SBATCH --mem=1000
#SBATCH --job-name=irace_job
#SBATCH --cpus-per-task=16

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 TARGET_FILE OUTFILE"
    exit 1
fi

# Read command line arguments
TARGET_DIRECTORY="$1"
OUTFILE="$2"

exec 5>&1
CMDOUT=$(enroot start -r -m "${TARGET_DIRECTORY}":/experiment automode bash -c "cd /experiment && irace --parallel 16" | tee >(cat - >&5))

line=$(echo "$CMDOUT" | grep -- '--nroot' | head -n 1)
result=${line#* }
dname=$(basename $TARGET_DIRECTORY)
echo "$dname;$result" >> "${OUTFILE}"

