# Produit matrice-vecteur v = A.u
import numpy as np
from time import time
from mpi4py import MPI

# Dimension du problème (peut-être changé)
dim = 1200
# Initialisation de la matrice
A = np.array([ [(i+j)%dim+1. for i in range(dim)] for j in range(dim) ])
#print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
#print(f"u = {u}")


#To verify that the rsult is correct




comGlobal=MPI.COMM_WORLD.Dup()
rank=comGlobal.rank
nbp=comGlobal.size
vVerif = A.dot(u)


if dim%nbp!=0:
    print(f"Must have a number of processes which divides the dimension {dim} of the vectors")
    comGlobal.Abort(-1)

NLoc=dim//nbp
jbeg=rank*NLoc
jend=(rank+1)*NLoc

if(rank==0):
    print(f"Nloc={NLoc}")
    v=np.empty(dim, dtype=np.double)
else :
    v=np.empty(dim, dtype=np.double)


ALoc=A[:,jbeg:jend]
debpara=time()
vLoc=ALoc.dot(u)
finpara=time()
print(f"Temps de calcul pour le thread {rank} : {(finpara-debpara)*1000} ms")


comGlobal.Gather(vLoc,v, root=0)
comGlobal.Bcast([v, MPI.DOUBLE], root=0)
comGlobal.Bcast([vVerif, MPI.DOUBLE], root=0)


if(np.array_equal(v,vVerif)):
    print(f"Processus {rank}: bonne valeur du produit reçue")
else :
    print(f"Processus {rank}: mauvaise valeur du produit reçue")


