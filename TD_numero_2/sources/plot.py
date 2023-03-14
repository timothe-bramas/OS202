import matplotlib.pyplot as plt

x=[1,2,4,8,16]
y=[1,1.8,1.45,0.075,0.009]

plt.plot(x,y)
plt.xlabel("Nombre de processeurs")
plt.ylabel("Speed-up")
plt.title("Speed-up du produit matrice-vecteur")
plt.show()