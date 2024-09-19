export n=32
export np=4

mpirun -n $np python3 forced_convection_2D.py --nx $n --nz $n --nproc $np --order 3
mpirun -n 1 python3 forced_convection_2D.py --nx $n --nz $n --nproc $np --order 3 --plot
#mpirun -n $np python3 energy_forced_convection_2D.py --nx $n --nz $n --nproc $np --order 3
#mpirun -n 1 python3 energy_forced_convection_2D.py --nx $n --nz $n --nproc $np --order 3 --plot