#!/bin/bash

#PBS -P dp9
#PBS -q normalsr
#PBS -l walltime=1:00:00
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l storage=gdata/hh5+gdata/tm70+gdata/ik11+scratch/kr97
#PBS -l wd
#PBS -j oe

module use /g/data/hh5/public/modules
module load conda/analysis3
module load openmpi

export n=32
export np=16

mpirun -n $np python3 video_forced_convection_2D.py --nx $n --nz $n --nproc $np --order 3
mpirun -n 8 python3 video_forced_convection_2D.py --nx $n --nz $n --nproc $np --order 3 --plot
#mpirun -n $np python3 energy_forced_convection_2D.py --nx $n --nz $n --nproc $np --order 3
#mpirun -n 1 python3 energy_forced_convection_2D.py --nx $n --nz $n --nproc $np --order 3 --plot