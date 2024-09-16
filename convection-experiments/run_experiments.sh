mpirun -n 4 python3 forced_convection_2D.py --nx 32 --nz 32 --nproc 4 --order 3
mpirun -n 1 python3 forced_convection_2D.py --nx 32 --nz 32 --nproc 4 --order 3 --plot