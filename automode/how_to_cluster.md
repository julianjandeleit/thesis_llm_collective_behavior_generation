# how to docker (podman) to enroot with cluster
## on local machine
make sure the most recent version of automode is built according to readme.

enroot import --output automode.sqsh podman://automode # replace podman with dockerd for docker
scp automode.sqsh  kn_pop515691@bwunicluster.scc.kit.edu:~/automode.sqsh

optional:
compress for transmission:
zstd -i automode.sqsh -o automode.zstd

## on cluster

enroot create -n automode automode.sqsh
enroot start automode ls # example command

### actually perform simulation
enroot start -m aggregation.argos:/root/aac.argos automode AutoMoDe/bin/automode_main_bt -c aac.argos --bt-config --nroot 3 --nchildroot 1 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.26 --n01 5 --a01 0 --rwm01 5 --p01 0 | sed '/STOP/d'

### slurm

create a job script with metadata:

#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=1
#SBATCH --mem=1000
#SBATCH --job-name=testjob

enroot start -m aggregation.argos:/root/aac.argos automode AutoMoDe/bin/automode_main_bt -c aac.argos --bt-config --nroot 3 --nchildroot 1 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.26 --n01 5 --a01 0 --rwm01 5 --p01 0 | sed '/STOP/d' > outfile


show available partitions:

sinfo_t_idle

submit partition:

sbatch --partition dev_single testscript.sh

show running jobs:
squeue
show details of job
scontrol show job ID

output gets written to slurm-JOBID.out


