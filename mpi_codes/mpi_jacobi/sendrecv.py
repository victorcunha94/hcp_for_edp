import sys
import numpy as np
from mpi4py import MPI


comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

if size<2:
   print(f"Este programa trabalha apenas para o último processo. ")
