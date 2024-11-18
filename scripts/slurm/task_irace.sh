#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --mem=500
#SBATCH --job-name=irace_job
#SBATCH --cpus-per-task=4

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 TARGET_FILE OUTFILE"
    exit 1
fi

# Read command line arguments
TARGET_DIRECTORY="$1"
OUTFILE="$2"

CMDOUT=$(enroot start -r -m experiment:/experiment automode bash -c "cd /${TARGET_DIRECTORY} && irace")

line=$(echo "$CMDOUT" | grep -- '--nroot' | head -n 1)
result=${line#* }
echo "$TARGET_DIRECTORY;$result" >> "${OUTFILE}"

