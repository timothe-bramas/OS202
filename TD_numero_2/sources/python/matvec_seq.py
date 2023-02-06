# Produit matrice-vecteur v = A.u
import numpy as np
from time import time
# Dimension du problème (peut-être changé)
dim = 1200
# Initialisation de la matrice
A = np.array([ [(i+j)%dim+1. for i in range(dim)] for j in range(dim) ])
#print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
#print(f"u = {u}")

# Produit matrice-vecteur
deb=time()
v = A.dot(u)
fin=time()
#print(f"v = {v}")
print(f"Temps de calcul en séquentiel : {fin-deb}")
