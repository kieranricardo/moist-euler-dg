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

export nz=32
export np=16
mpirun -n $np python3 ice_moist_bubble_2D.py --n $nz --nproc $np
python3 ice_moist_bubble_2D.py --n $nz --nproc $np --plot

export nz=64
export np=32
mpirun -n $np python3 ice_moist_bubble_2D.py --n $nz --nproc $np
python3 ice_moist_bubble_2D.py --n $nz --nproc $np --plot

export nz=128
export np=64
mpirun -n $np python3 ice_moist_bubble_2D.py --n $nz --nproc $np
python3 ice_moist_bubble_2D.py --n $nz --nproc $np --plot

export nz=200
export np=100
mpirun -n $np python3 ice_moist_bubble_2D.py --n $nz --nproc $np
python3 ice_moist_bubble_2D.py --n $nz --nproc $np --plot