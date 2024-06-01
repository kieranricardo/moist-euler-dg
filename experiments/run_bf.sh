#!/bin/bash

#PBS -P dp9
#PBS -q normalsr
#PBS -l walltime=1:00:00
#PBS -l ncpus=104
#PBS -l mem=400GB
#PBS -l storage=gdata/hh5+gdata/tm70+gdata/ik11+scratch/kr97
#PBS -l wd
#PBS -j oe

module use /g/data/hh5/public/modules
module load conda/analysis3
module load openmpi

export nz=64
# export np=$(( 2*$nz ))
export np=$(( $nz / 2))

# mpirun -n $np python3 dry_bubble_2D.py --n $nz --nproc $np

mpirun -n $np python3 moist_bubble_2D.py --n $nz --nproc $np
python3 moist_bubble_2D.py --n $nz --nproc $np --plot
