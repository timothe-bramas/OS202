#Utilisation de ghost cells : 
#En répartissant les lignes entre les différents processus de calcul on leur donne une ligne de plus
#pour qu'ils puissent faire le calcul en fonction du plus proche voisin. Mais calcul effectué sur NLoc-1 lignes
#Il faudra alors récupérer à chaque itération les nouvelles ghost cells, mises à jour par le processus voisin
#Idée : chaque processus a un tableau diffloc, un tableau ghostcells


"""
Le jeu de la vie
################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonales.) L'interaction se fait selon les
règles suivantes pour calculer l'irération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

On itère ensuite pour étudier la façon dont évolue la population des cellules sur la grille.
"""
import tkinter as tk
import numpy   as np

class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    En entrée lors de la création de la grille :
        - dimensions est un tuple contenant le nombre de cellules dans les deux directions (nombre lignes, nombre colonnes)
        - init_pattern est une liste de cellules initialement vivantes sur cette grille (les autres sont considérées comme mortes)
        - color_life est la couleur dans laquelle on affiche une cellule vivante
        - color_dead est la couleur dans laquelle on affiche une cellule morte
    Si aucun pattern n'est donné, on tire au hasard quels sont les cellules vivantes et les cellules mortes
    Exemple :
       grid = Grille( (10,10), init_pattern=[(2,2),(0,2),(4,2),(2,0),(2,4)], color_life="red", color_dead="black")
    """
    def __init__(self, dim, init_pattern=None, color_life="black", color_dead="white"):
        import random
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i,indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self, extracells=None, ybeg=0, yend=0):

        ny = self.dimensions[0]
        nx = self.dimensions[1]
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        diff_cells = []
        #Met à jour les premières lignes
        for i in range(ny-1):
            i_above = (i+ny-1)%ny
            i_below = (i+1)%ny
            for j in range(nx):
                j_left = (j-1+nx)%nx
                j_right= (j+1)%nx
                voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below]
                voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right]
                voisines = np.array(self.cells[voisins_i,voisins_j])
                nb_voisines_vivantes = np.sum(voisines)
                if self.cells[i,j] == 1: # Si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[i,j] = 0 # Cas de sous ou sur population, la cellule meurt
                        diff_cells.append((i+ybeg)*nx+j)
                    else:
                        next_cells[i,j] = 1 # Sinon elle reste vivante
                elif nb_voisines_vivantes == 3: # Cas où cellule morte mais entourée exactement de trois vivantes
                    next_cells[i,j] = 1         # Naissance de la cellule
                    diff_cells.append((i+ybeg)*nx+j)
                else:
                    next_cells[i,j] = 0         # Morte, elle reste morte.
            
        #Met à jour la dernière ligne
        i=ny-1
        for j in range(nx) :

            j_left = (j-1+nx)%nx
            j_right = (j+1)%nx
            i_above =  ny-2

            voisins_i=[i_above,i_above,i_above,i,i]
            voisins_j=[j_left,j,j_right,j_left,j_right]
            idx_botoms=[j_left,j,j_right]
            voisines_bottom=np.array(extracells[idx_botoms])
            voisines=np.array(self.cells[voisins_i,voisins_j])
            nb_voisines_vivantes=np.sum(voisines)+np.sum(voisines_bottom)
            if self.cells[i,j]==1:
                if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                    next_cells[i,j] = 0 # Cas de sous ou sur population, la cellule meurt
                    diff_cells.append((i+ybeg)*nx+j)
                else :
                    next_cells[i,j] = 1 #la cellule reste vivante, pas de diff
            elif nb_voisines_vivantes == 3:
                next_cells[i,j] =1
                diff_cells.append((i+ybeg)*nx+j)
            else : 
                next_cells[i,j] =0
        return diff_cells




    def update(self, diff_cells):
        "Mise à jour du tableau des cellules"
        nx = self.dimensions[1]
        for ind in diff_cells:
            if self.cells[ind//nx, ind%nx] == 1:
                self.cells[ind//nx, ind%nx] = 0
            else:
                self.cells[ind//nx, ind%nx] = 1


class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran
        - geometry est un tuple de deux entiers donnant le nombre de pixels verticaux et horizontaux (dans cet ordre)
        - grid est la grille décrivant l'automate cellulaire (voir plus haut)
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        # Calcul de la taille d'une cellule par rapport à la taille de la fenêtre et de la grille à afficher :
        self.size_x = geometry[1]//grid.dimensions[1]
        self.size_y = geometry[0]//grid.dimensions[0]
        if self.size_x > 4 and self.size_y > 4 :
            self.draw_color='lightgrey'
        else:
            self.draw_color=""
        # Ajustement de la taille de la fenêtre pour bien fitter la dimension de la grille
        self.width = grid.dimensions[1] * self.size_x
        self.height= grid.dimensions[0] * self.size_y
        # Création de la fenêtre à l'aide de tkinter
        self.root = tk.Tk()
        # Création de l'objet d'affichage
        self.canvas = tk.Canvas(self.root, height=self.height, width=self.width)
        self.canvas.pack()
        #
        self.canvas_cells = []

    def compute_rectange(self, i: int, j: int):
        """
        Calcul la géométrie du rectangle correspondant à la cellule (i,j)
        """
        return (self.size_x*j, self.height - self.size_y*i - 1,
                self.size_x*j+self.size_x-1, self.height - self.size_y*(i+1) )

    def compute_color(self, i: int, j: int):
        if self.grid.cells[i,j] == 0:
            return self.grid.col_dead
        else:
            return self.grid.col_life

    def draw(self, diff):
        if len(self.canvas_cells) == 0:
            # Création la première fois des cellules en tant qu'entité graphique :
            self.canvas_cells = [ self.canvas.create_rectangle(*self.compute_rectange(i,j), fill=self.compute_color(i,j),
                                       outline=self.draw_color) for i in range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]
        else:
            nx = grid.dimensions[1]
            ny = grid.dimensions[0]
            [self.canvas.itemconfig(self.canvas_cells[ind], fill=self.compute_color(ind//nx, ind%nx), outline=self.draw_color) for ind in diff]
        self.root.update_idletasks()
        self.root.update()

if __name__ == '__main__':
    import time
    import sys
    from mpi4py import MPI

    GlobComm = MPI.COMM_WORLD.Dup()
    nbp     = GlobComm.size
    rank    = GlobComm.rank

    if(nbp<2) :
        print(f"Must have at least 2 processors")
        GlobComm.Abort(-1)
    dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((200,100),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }
    choice = 'space_ship'
    if len(sys.argv) > 1 :
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3 :
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    if rank == 0:
        print(f"Pattern initial choisi : {choice}")
        print(f"resolution ecran : {resx,resy}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)
    grid = Grille(*init_pattern)

    if rank == 0:
        appli = App((resx, resy), grid)
        numdiffloc=0
        diffloc=[]

    #Calcul du nombre de lignes qu'aura à calculer chaque thread
    else :
        extra=init_pattern[0][0]%(nbp-1)
        Nloc=init_pattern[0][0]//(nbp-1)
 
        if(rank<nbp-extra):
            ybeg=(rank-1)*Nloc
            yend=(rank)*Nloc-1
        else : 
            ybeg=(nbp-extra-1)*Nloc+(rank-nbp+extra)*(Nloc+1)
            yend=ybeg+Nloc
            if(rank==nbp-1):
                yend=init_pattern[0][0]-1
            Nloc+=1
        
        #Extraline permettra d'actualiser les cellules de la dernière ligne de chaque thread
        if(rank<nbp-1):
            next_thread=rank+1
            extracells=grid.cells[yend+1]
            if(rank==1):
                prev_thread=nbp-1
            else :
                prev_thread=rank-1
        else :
            extracells=grid.cells[0]
            next_thread=1
            prev_thread=rank-1
        loccells=grid.cells[ybeg:yend+1]
        gridloc=Grille((Nloc,init_pattern[0][1]))
        gridloc.cells=loccells


    while(True):
        tc1=time.time()
        #time.sleep(0.5) # A régler ou commenter pour vitesse maxi
        if rank != 0:
            diffloc = gridloc.compute_next_iteration(extracells,ybeg,yend)
            GlobComm.send(gridloc.cells[0],dest=next_thread, tag=1234+rank)
            extracells=GlobComm.recv(source=prev_thread, tag=1234+prev_thread) 
            numdiffloc=np.sum(np.size(diffloc))
        
        #Définition du nombre de changements de chaque thread
        numdiff = np.array(GlobComm.gather(numdiffloc, root=0))
        if(rank!=0):
            numdiff=np.empty(nbp,dtype=int)
        GlobComm.Bcast(numdiff,root=0)

        if(rank==0):
            diff=np.empty(np.sum(numdiff), dtype=int)
        else :
            diff=None

        GlobComm.Gatherv(diffloc,[diff,numdiff,[0,0,0,0],MPI.INT],root=0)

        if rank == 0:
            ta1 = time.time()
            tc2=time.time()
            grid.update(diff)
            appli.draw(diff) 
            ta2 = time.time()
            print(f"Temps calcul prochaine generation : {tc2-tc1:2.2e} secondes, temps affichage : {ta2-ta1:2.2e} secondes\r", end='');

