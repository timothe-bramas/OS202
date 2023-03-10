# Calcul pi par une méthode stochastique (convergence très lente !)
import time
import numpy as np
import mpi4py



globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank


# Nombre d'échantillons :
nbSamples = 40000000

beg = time.time()
# Tirage des points (x,y) tirés dans un carré [-1;1] x [-1; 1]

if(rank==0) :
    x = 2.*np.random.random_sample((nbSamples,))-1.
    globCom.send(rank,x,1)

elif(rank==1):
    y = 2.*np.random.random_sample((nbSamples,))-1.

    # Création masque pour les points dans le cercle unité
    filtre = np.array(x*x+y*y<1.)

    # Compte le nombre de points dans le cercle unité
    sum = np.add.reduce(filtre, 0)

    approx_pi = 4.*sum/nbSamples
    end = time.time()
    print(f"Temps pour calculer pi : {end - beg} secondes")
    print(f"Pi vaut environ {approx_pi}")

