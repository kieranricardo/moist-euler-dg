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

export nz=104
export np=$nz

# mpirun -n $np python3 dry_bubble_2D.py --n $nz --nproc $np

mpirun -n $np python3 moist_bubble_2D.py --n $nz --nproc $np --o 3
mpirun -n $np python3 moist_bubble_2D.py --n $nz --nproc $np --o 4
mpirun -n $np python3 moist_bubble_2D.py --n $nz --nproc $np --o 5
mpirun -n $np python3 moist_bubble_2D.py --n $nz --nproc $np --o 6

python3 moist_bubble_2D.py --n $nz --nproc $np --plot --o 3
python3 moist_bubble_2D.py --n $nz --nproc $np --plot --o 4
python3 moist_bubble_2D.py --n $nz --nproc $np --plot --o 5
python3 moist_bubble_2D.py --n $nz --nproc $np --plot --o 6
