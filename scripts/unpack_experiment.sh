#!/bin/bash
  
# Check if a command-line argument is provided, otherwise use default
NEW_EXP_NAME=${1:-newexp}

rm -r $TMPDIR/irace_swarm
mkdir -p $TMPDIR/irace_swarm
mv irace_experiments_slurm.zip $TMPDIR/irace_swarm/
cd $TMPDIR/irace_swarm
unzip irace_experiments_slurm.zip
rm irace_experiments_slurm.zip
#rm -r ~/generated_irace_datasets
# cp -r generated_irace_datasets ~
chmod +x generated_irace_datasets/task_irace.sh
cp -r generated_irace_datasets ~/$NEW_EXP_NAME
cd ~
echo "experiments prepared in ~/$NEW_EXP_NAME"
# echo "data in $TMPDIR/irace_swarm"
# ls $TMPDIR/irace_swarm
