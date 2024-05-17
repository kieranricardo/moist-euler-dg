#!/bin/bash

#PBS -P dp9
#PBS -q expresssr
#PBS -l walltime=0:10:00
#PBS -l ncpus=104
#PBS -l mem=140GB
#PBS -l storage=gdata/hh5+gdata/tm70+gdata/ik11+scratch/kr97
#PBS -l wd

module use /g/data/hh5/public/modules
module load conda/analysis3
module load openmpi

export nz=40
export np=$(( 2*$nz ))

mpirun -n $np python3 dry_bubble_2D.py --n $nz
mpirun -n $np python3 moist_bubble_2D.py --n $nz
mpirun -n $np python3 ice_moist_bubble_2D.py --n $nz
mpirun -n $np python3 no_ice_moist_bubble_2D.py --n $nz
