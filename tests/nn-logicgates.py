#dependances et reglages
import numpy as np
import matplotlib.pyplot as plt
from LibNeuralNetsigm import *
np.set_printoptions(precision=1)

#base de donnees
datain = normalize(np.array([[0,0,0],
                             [0,0,1],
                             [0,1,0],
                             [0,1,1],
                             [1,0,0],
                             [1,0,1],
                             [1,1,0],
                             [1,1,1]],dtype = np.float64))                    #valeurs d'entree
dataout = normalize(np.array([[0,0,1],
                              [0,1,0],
                              [0,1,1],
                              [1,0,0],
                              [1,0,1],
                              [1,1,0],
                              [1,1,1],
                              [0,0,0]],dtype = np.float64))                   #valeurs de sortie
# datain = normalize(np.array([[0,0],
#                              [0,1],
#                              [1,0],
#                              [1,1]],dtype = np.float64))                        #valeurs d'entree
# dataout = normalize(np.array([[0,1,0,0],
#                               [0,1,1,1],
#                               [0,1,1,1],
#                               [1,0,1,0]],dtype = np.float64))                       #valeurs de sortie

#parametres de l'apprentissage
nbruns = 1                                                                      #nombre de reseaux a entrainer
epochs = 9999                                                                   #nombre d'etapes d'entrainement
prec = 1                                                                        #precision de l'entrainement
err = [[] for i in range((len(dataout[0]) * len(datain))+1)]                        #tableau des erreurs au cours du temps de chacun des reseaux

datain = normalize(datain)
dataout = normalize(dataout)
# affichage de la base de donnees
print "\ndata ="
for i in range(len(datain)):
    print datain[i],dataout[i]

#apprentissage et evaluation du reseau
N = NeuralNet([len(datain[0]),5,len(dataout[0])])
for epoch in range(epochs):
    totalerror = 0
    for i in range(len(datain)):
        N.forward(datain[i])
        N.backprop(dataout[i],prec)
        error = N.error(dataout[i])
        totalerror += np.sum(error)
        # print error,range(len(dataout[0]))
        for out in range(len(dataout[0])):
            err[out+i*(len(dataout[0]))+1].append(error[out])
            # print err[out]

    err[0].append(totalerror)
    # err[len(datain)*len(dataout[0])].append(0)

for i in range(len(datain)):
    for out in range(len(dataout[0])):
        plt.plot(err[out+i*len(dataout[0])])

for i in range(len(datain)):
    N.forward(datain[i])
    print "input : \n",datain[i],"\nprediction : \n",N.O[N.numberOfLayers-1].T[0]
    print "error : \n",np.sum(N.error(dataout[i])),"\n"

N.save()
N.show(True,True)
print(np.array(err).T)
plt.ylabel("Erreur")
# plt.show()
