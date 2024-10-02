#!/bin/bash

#PBS -P dp9
#PBS -q normalsr
#PBS -l walltime=0:20:00
#PBS -l ncpus=104
#PBS -l mem=400GB
#PBS -l storage=gdata/hh5+gdata/tm70+gdata/ik11+scratch/kr97
#PBS -l wd
#PBS -j oe

module use /g/data/hh5/public/modules
module load conda/analysis3
module load openmpi

declare -a arr=(8 16 32 64 128)
export o=3
export np=4

for nz in ${arr[@]}
do
   mpirun -n $np python3 ice_moist_bubble_2D.py --n $nz --nproc $np --o $o
   python3 ice_moist_bubble_2D.py --n $nz --nproc $np --plot  --o $o

done
