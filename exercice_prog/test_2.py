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
y[41:] = 2000  / 10000

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
nc = 100
nw = 20  # nombre de points sur la grille
wgrid = np.linspace(0, np.max(y) * 20, nw)
cgrid = np.linspace(np.max(y)/20, np.max(y) * 10, nc)


# dans le problème avec interpolation il faut
# également créer une grille de consommation

# on définit deux grilles
# une où on va enregistrer les valeurs prises par la fonction valeur
# pour différents t et wt
# une où on va enregistrer les valeurs prises par la fonction de consommation
# pour différents t et wt
Vfin = np.zeros((Tmax + 2, nw))
cfin = np.zeros((Tmax + 1, nw))

#@jit(nopython=True)
def find_policies(V, cpol):
    # V est la fonction valeur
    # cpol est la fonction de consommation
    for i in range(Tmax + 1):
        t = Tmax - i  # on commence par la dernière période
        for j in range(nw-1):  # boucle à travers wt
            wt = wgrid[j]
            Vtemp = -10**9  # initialisation de la valeur en wt à un niveau très bas
            ctemp = 0
            for k in range(nc):  # boucle à travers c(t)
                ct = cgrid[k]
                wtp1 = R*(wt+y[t]-ct) #calcul v(t+1)

                if wtp1 >= 0:

                    for m in range(nw-1): #Boucle à travers tous les v(t+1) de ma grille
                        if wtp1 >= wgrid[m] and wtp1 <= wgrid[m+1]:
                            #Alors nous interpellons
                            m, b = equation_droite(V[t+1, m+1], V[t+1, m], wgrid[m+1], wgrid[m])
                            Vtemp2 = util(ct) + beta * (b+m*wtp1)
                




                ct = wt + y[t] - wtp1 / R  # calcul de ct
                Vtemp2 = util(ct) + beta * V[t + 1, k] #Utilité de la consommation de la période courante
                if ct > 0 and Vtemp2 > Vtemp:
                          # la remplacer par Vtemp2  # si ct positif caculer l'utilité espérée et si cette utilité espérée est plus grande que Vtemp                   
                        Vtemp = Vtemp2  # la remplacer par Vtemp2
                        ctemp = ct     # remplacer ctemp par ct

                else:
                    Vtemp = Vtemp # 

            V[t, j] = Vtemp
            #Tracer()()  # une fois qu'on a fait la boucle à travers tous
            # les w(t+1) mettre à jour la valeur prise par la fonction valeur
            # de la période t au point wt
            cpol[t, j] = ctemp  # mettre à jour également la règle de décision

    return V, cpol

Vfin, cfin = find_policies(Vfin, cfin)

# graphique fonction valeur
plt.plot(wgrid,Vfin[0])
plt.show()

# graphique règle de décision
plt.plot(wgrid,cfin[0])
plt.plot(wgrid,cfin[40])
plt.show()


csimul = np.zeros(Tmax + 1)  # consommation simulée
wsimul = np.zeros(Tmax + 2)  # richesse simulée
wsimul[0] = 30000 / 10000  # richesse initiale

for t in range(Tmax + 1):
    wt = wsimul[t]
    for i in range(nw - 1):
        # regarder où wt se situe dans la grille
        if wt >= wgrid[i] and wt < wgrid[i + 1]:
            # afin de déterminer entre quels poinst on interpole

            b = (cfin[t, i + 1] - cfin[t, i]) / \
                (wgrid[i + 1] - wgrid[i])  # calcul de la pente
            # calcul de l'ordonnée à l'origine
            a = cfin[t, i + 1] - b * wgrid[i + 1]

            ct = a + b * wt  # consommation en t quand richesse égale à wt

            wtp1 = R * (y[t] + wt - ct)  # calcul de w(t+1)
            # on enregistre w(t+1) qui sera utilisé dans la boucle suivante
            wsimul[t + 1] = wtp1
            csimul[t] = ct  # on enregistre ct

            break

# graphique de la consommation et de la richesse
plt.plot(wsimul * 10000)
plt.plot(csimul * 10000, '--')
plt.plot((40, 40), (0, np.max(wsimul) * 10000 * 1.1),
         'k-')  # ligne montrant l'âge de la retraite
plt.show()
