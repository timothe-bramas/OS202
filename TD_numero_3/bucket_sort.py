from mpi4py import MPI
import numpy as np
import time
import sys

#Pour regrouper les bouts de liste venant de tous les vecteurs de différentes tailles : gatherV
def findBucket(x,numproc,bornessup):
    for i in range(numproc):
        if(x<=bornessup[i]):
            return i

globCom=MPI.COMM_WORLD.Dup()
nbp=globCom.size
rank=globCom.rank


N=32

if len(sys.argv) > 1:
    N = int(sys.argv[1])


reste=N%nbp
assert reste==0
Nloc=N//nbp
numbornesloc=nbp+1

#Répartition du vecteur à trier
if(rank==0):
    values=np.random.randint(-32768,32768,size=N,dtype=np.int64)
    bornes=np.empty(nbp*numbornesloc, dtype=np.int64)

else :
    values=None
    bornes=None

valuesloc=np.empty(Nloc,dtype=np.int64)


globCom.Scatter([values,MPI.INT64_T], [valuesloc, MPI.INT64_T], root=0)

#Tri local dans chaque processus pour définir les bornes

valuesloc=np.sort(valuesloc)

bornesloc=np.empty(numbornesloc, dtype=np.int64)
for ii in range(numbornesloc-1):
    bornesloc[ii] = valuesloc[ii*Nloc//numbornesloc]
bornesloc[numbornesloc-1] = valuesloc[Nloc-1]

#print(f"Taille bornesloc {np.size(bornesloc)}")
#print(f"Bornes:  b1 : {bornesloc[0]}, b2 : {bornesloc[2]}, b3 : {bornesloc[numbornesloc-1]}, \n")



#if(rank==0):
    #print(f"Taille bornes {np.size(bornes)}")

globCom.Gather(bornesloc,bornes, root=0)

#if(rank==0):
#   print(f"Bornes : {bornes}\n")

#Répartition des seaux à chaque thread

if(rank==0):
    bornes=np.sort(bornes)
    bornes=np.resize(bornes, (nbp,numbornesloc))
    bornessup=bornes[:,numbornesloc-1]
    bucketcount=np.zeros(nbp, np.int64)
    buckets=np.zeros(N,np.int64)
    for i in range(N) :
        bucket=findBucket(values[i],nbp,bornessup)
        buckets[i]=bucket 
        bucketcount[bucket]+=1
    bucketsize=np.empty(1, np.int64)
    #print(f"Taille des seaux : {bucketcount}\n")  

#Remplacer cette partie par un Gather fait par chaque thread, récupérant les bons 
#éléments donnés # par chaque autre thread, calculés avec find_bucket après un scatter des bornes




else :
    bucketsize=np.empty(1, np.int64)
    bucketcount=None
    buckets=None

globCom.Scatter([bucketcount,MPI.INT64_T], [bucketsize, MPI.INT64_T], root=0)
bucketsize=bucketsize[0]
Bucket=np.empty(bucketsize,np.int64)
#print(f"Taille du seau : {Bucket.size}\n")  

#if(rank==0):
#    print(f"Taille des données : {buckets.size}")
#    print(f"buckets : {buckets}")

