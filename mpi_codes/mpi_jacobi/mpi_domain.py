from mpi4py import MPI
import numpy as np



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
   print(f"Eu sou 1 dentre {size} processos ")


print(f"Oi, eu sou o processo {rank}")

