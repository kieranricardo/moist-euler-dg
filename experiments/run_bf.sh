#!/bin/bash

#PBS -P dp9
#PBS -q normalsr
#PBS -l walltime=0:30:00
#PBS -l ncpus=104
#PBS -l mem=400GB
#PBS -l storage=gdata/xp65+gdata/tm70+gdata/ik11+scratch/kr97
#PBS -l wd
#PBS -j oe

module use /g/data/xp65/public/modules
module load conda/analysis3-25.09

export nz=64
export np=$(( $nz / 2))

# mpirun -n $np python3 dry_bubble_2D.py --n $nz --nproc $np

mpirun -n $np python3 moist_bubble_2D.py --n $nz --nproc $np --o 3
python3 moist_bubble_2D.py --n $nz --nproc $np  --o 3 --plot
