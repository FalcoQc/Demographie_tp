import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from IPython.core.debugger import Tracer
#from time import time

# nous fixons les paramètres du modèle
gamma = 2.0
beta = 0.97
R = 1.01
Tmax = 60

# le revenu n'est pas supposé constant au cours du temps
y = np.zeros(Tmax + 1)
# revenu égal à 15000 pour les périodes 0 à 40 et ensuite 0 ("retraite")
y[:41] = 15000 / 10000

# nous définissons une fonction qui nous donne
# l'utilité pour un niveau de consommation c


@jit(nopython=True)
def util(c):
    if gamma == 1.0:
        return np.log(c)
    else:
        return c**(1 - gamma) / (1 - gamma)

def equation_droite(y2, y1, x2, x1):
	m = (y2 - y1) / (x2 - x1) #Pente
	b = y2 - m * x2 #Ordonné à l'origine
	return m, b


	


# nous définissons une grille de richesse
nw = 100  # nombre de points sur la grille
nc = 100 # nombre de points sur la grille de consommation
wgrid = np.linspace(0, np.max(y) * 20, nw)
cgrid = np.linspace(np.max(y)/20, np.max(y) * 10, nc)

# dans le problème avec interpolation il faut
# également créer une grille de consommation

# on définit deux grilles
# une où on va enregistrer les valeurs prises par la fonction valeur
# pour différents t et wt
# une où on va enregistrer les valeurs prises par la fonction de consommation
# pour différents t et wt
Vfin = np.zeros((Tmax + 3, nw))
cfin = np.zeros((Tmax + 2, nc))


def find_policies(V, cpol):
	compteur = 0
	compteur_2 = 0
	#V est la fonction de valeur
	#cpol est la fonction de consommation
	for i in range(Tmax+1):
		t = Tmax - i # On commence par la dernière période
		Vtemp = - 10 ** 9
		ctemp = 0
		for j in range(nw - 1): #Boucle à travers wt
			wt = wgrid[j]
			wt_2 = wgrid[j+1] #Pour trouver le point suivant
			for k in range(nc-1): #Boucle à tracers ct qui modifie wt+1 (Je crois...)
				ct = cgrid[k]
				wtp1 =   R * (wt   + y[t] - ct)
				if wtp1 > 0 and wtp1 < wgrid[nw-1]: #Nous allons interpoler entre w_i et w_ip1
					for w in range(nw-2):
						if wtp1 >= wgrid[w] and wtp1 < wgrid[w+1]:
							m, b = equation_droite(wgrid[w+1], wgrid[w], w+1, w) 
							#Je ne comprends pas quel point je dois utiliser pour l'interpolation
							#Comment trouver l'équivalent pour la foncrion de valeur d'un wt
							Vtemp2 = m * ct + b
							if ct > 0 and Vtemp2 > Vtemp:
								Vtemp = Vtemp2
								ctemp = ct
				elif wtp1 > wgrid[nw - 1]:
					pass
					#Nous allons extrapoler

				else:
					pass
							

				

				V[t, k] = Vtemp
					
				cpol[t, k] = ctemp

		
	return(V, cpol)

Vfin, cfin = find_policies(Vfin, cfin)

# graphique fonction valeur
plt.plot(wgrid,Vfin[60])
plt.show()






