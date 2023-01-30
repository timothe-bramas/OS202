import mpi4py
import sys

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank

if(rank==0):
    token=1

if(rank<nbp):
    globCom.send(rank,token,rank+1)
else :
    print(token)