import numpy as np
import matplotlib.pyplot as plt
from numba import jit
#from time import time

# nous fixons les paramètres du modèle
gamma = 2.0
beta  = 0.97
R     = 1.01
Tmax  = 60 

# le revenu n'est pas supposé constant au cours du temps
y      = np.zeros(Tmax+1)
y[:41] = 15000/10000 # revenu égal à 15000 pour les périodes 0 à 40 et ensuite 0 ("retraite")
y[41:] = 2000/10000
# nous définissons une fonction qui nous donne 
# l'utilité pour un niveau de consommation c
@jit(nopython=True)
def util(c):
    if gamma == 1.0:
        return np.log(c)
    else:
        return c**(1-gamma) / (1-gamma)
    
        
# nous définissons une grille de richesse
nw = 60 # nombre de points sur la grille
wgrid = np.linspace(0,np.max(y)*20,nw)

# nous définissons une grille de consommation
nc = 300 # nombre de points sur la grille
cgrid = np.linspace(np.max(y)/20,np.max(y)*10,nc)

# dans le problème avec interpolation il faut 
# également créer une grille de consommation

# on définit deux grilles
# une où on va enregistrer les valeurs prises par la fonction valeur
# pour différents t et wt
# une où on va enregistrer les valeurs prises par la fonction de consommation
# pour différents t et wt
Vfin    = np.zeros((Tmax+2,nw))
cfin = np.zeros((Tmax+1,nw))

@jit(nopython=True)
def find_policies(V,cpol):
    # V est la fonction valeur
    # cpol est la fonction de consommation
    for i in range(Tmax+1):
        t = Tmax-i # on commence par la dernière période 
        for j in range(nw): # boucle à travers wt
            wt = wgrid[j] 
            Vtemp = -10**9 # initialisation de la valeur en wt à un niveau très bas
            ctemp = 0      
            for k in range(nc): # boucle à travers c(t)
                ct   = cgrid[k] # calcul de ct
                wtp1 = R*(wt+y[t]-ct) # calcul w(t+1)
                if wtp1 >= 0: # si w(t+1) positif caculer l'utilité espérée
                    
                    
                    if wtp1 > wgrid[nw-1]:
                        # on extrapole 
                        b = (V[t+1,nw-1]-V[t+1,nw-2]) / (wgrid[nw-1]-wgrid[nw-2])
                        a = V[t+1,nw-1] - b*wgrid[nw-1]           
                        Vtemp2 = util(ct)+beta * (a+b*wtp1) #Donne l'utilité de la consommation en fonction de ct et wt+1
                    else:
                        for m in range(nw-1) : # boucle à travers tous les w(t+1) de la grille
                            if wtp1 >= wgrid[m] and wtp1 <= wgrid[m+1]:
                                # alors là on interpole
                                b = (V[t+1,m+1]-V[t+1,m]) / (wgrid[m+1]-wgrid[m])
                                a = V[t+1,m+1] - b*wgrid[m+1]           
                                Vtemp2 = util(ct)+beta * (a+b*wtp1)#Estimation de V[t+1, m] 
                                break # on arrête la boucle for si on a interpolé 
                                
                    # S'assurer que nous avons bel et bien un maximum
                    
                    if Vtemp2 > Vtemp: # si cette utilité espérée est plus grande que Vtemp
                        Vtemp = Vtemp2 # la remplacer par Vtemp2
                        ctemp = ct     # remplacer ctemp par ct
            
            V[t,j] = Vtemp  # une fois qu'on a fait la boucle à travers tous 
            # les w(t+1) mettre à jour la valeur prise par la fonction valeur 
            # de la période t au point wt 
            cpol[t,j] = ctemp # mettre à jour également la règle de décision
            
    return V, cpol

Vfin, cfin = find_policies(Vfin,cfin)

# graphique fonction valeur
plt.plot(wgrid,cfin[0])
plt.show() 