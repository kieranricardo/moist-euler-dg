#!/bin/bash

#PBS -P dp9
#PBS -q normalsr
#PBS -l walltime=0:20:00
#PBS -l ncpus=64
#PBS -l mem=100GB
#PBS -l storage=gdata/hh5+gdata/tm70+gdata/ik11+scratch/kr97
#PBS -l wd
#PBS -j oe

module use /g/data/hh5/public/modules
module load conda/analysis3
module load openmpi

declare -a arr=(8 16 32 64 128)
export o=3
# export np=4

for nz in ${arr[@]}
do
   export np=$(( $nz / 2))
   mpirun -n $np python3 smooth_moist_bubble_2D.py --n $nz --nproc $np --o $o
   python3 smooth_moist_bubble_2D.py --n $nz --nproc $np --plot  --o $o
done

 python3 convergence_ice_moist_bubble.py