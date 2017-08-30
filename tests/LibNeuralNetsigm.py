#dependances et reglages
import numpy as np
# import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

#L'objet reseau et ses methodes
class NeuralNet(object):
####Initialisation automatique du reseau
    def __init__(self, layers = [2,2]):
        self.numberOfLayers = len(layers)
        self.layers = layers
        self.I = []
        self.O = []
        self.E = []
        self.w = []
        self.Ew = []
        #Creation des vecteurs couche (entree I, sortie O, erreur E)
        for i in range(len(layers)):
            self.I.append(np.zeros(layers[i]).reshape((layers[i],1)))
            self.O.append(np.zeros(layers[i]).reshape((layers[i],1)))
            self.E.append(np.zeros(layers[i]).reshape((layers[i],1)))
        self.size = self.layers[self.numberOfLayers-1]
        #Creation des matrices de poid (poid w, erreur Ew)
        for i in range(self.numberOfLayers-1):
            self.w.append((np.random.rand(layers[i+1],layers[i])*2-1)/self.layers[i])
            self.Ew.append(np.zeros((layers[i+1],layers[i])))

####Methode permettant d'afficher les differentes valeurs du reseau pour le debuggage
    def show(self,showNeuron = False,showWeight = False):
        if showNeuron == True:
            print "size :",self.layers
            for i in range(len(self.layers)):
                print "I",i
                print self.I[i]
                print "O",i
                print self.O[i]
                print "E",i
                print self.E[i]
        if showWeight == True:
            for i in range(self.numberOfLayers-1):
                print "w",i
                print self.w[i]
            for i in range(self.numberOfLayers-1):
                print "Ew",i
                print self.Ew[i]

####ATTENTION les valeurs de sorties appartiennent a [0,1]
    def sigm(self,I,deriv=False):
        if deriv == True:
            s= self.sigm(I)
            return s*(1-s)
        else:
            return 1/(1+np.exp(-I))

####fonction d'erreur et sa derivee
    def error(self,Y,deriv=False):
        if deriv == True:
            return self.O[self.numberOfLayers-1]-Y##
        else:
            # print ((self.O[self.numberOfLayers-1]-Y)**2)/2
            return ((self.O[self.numberOfLayers-1].T[0]-Y)**2)/2

####methode d'utilisation du reseau, on donne I en entreeactualise le reseau avec les valeurs predites
    def forward(self,I):
        if len(I) != self.layers[0]:
            print "Your array",I,"is of size",len(I),"instead of",self.layers[0]
        else:
            I = np.asarray(I,dtype = np.float64).reshape((len(I),1))
            self.O[0] = I
            for c in range(self.numberOfLayers-1):
                self.I[c+1]=np.dot(self.w[c],self.O[c])+1
                self.O[c+1]=self.sigm(self.I[c+1])

####methode liee a l'apprentissage : optimise les poids par descente en gradient, fonction d'un exemple PRECEDEMMENT APPELE PAR forward, de sa valeur cible par rapport a l'erreur relative
    def backprop(self,Y,precision):
        if len(Y) != self.layers[self.numberOfLayers-1]:
            print "Your array",Y,"is of size",len(Y),"instead of",self.layers[self.size-1]
        else:
            Y = np.asarray(Y,dtype = np.float64).reshape((len(Y),1))
            if self.layers:
                #calcul des derivees relatives a la derniere couche
                self.E[self.numberOfLayers-1] = self.error(Y,True)*\
                     self.sigm(self.I[self.numberOfLayers-1],True)
                for c in range(self.numberOfLayers-2,-1,-1):
                    #calcul des derivees relatives aux couches precedentes
                    self.Ew[c] = np.dot(self.O[c],self.E[c+1].T)
                    self.w[c] -= self.Ew[c].T*precision
                    self.E[c] = self.sigm(self.I[c],True)*np.dot(self.w[c].T,self.E[c+1])
            else:
                print "Can't BackPropagate an empty Neural Net"

    def save(self,filename = "save.out"):
        counter = 1
        with file(filename, 'w') as outfile:
            outfile.write('# layers = \n')
            np.savetxt(outfile, np.asarray(self.layers).reshape(1,self.numberOfLayers), delimiter=',',fmt='%.16g')
            for layer in self.w:
                outfile.write('# layer: %d\n'%counter)
                np.savetxt(outfile, layer.reshape(1,np.prod(layer.shape)), delimiter=',',fmt='%.16g')
                counter += 1

    def load(self,filename = "save.out"):
        with file(filename, 'r') as F:
            array = []
            for line in F:
                if not line.strip().startswith("#"):
                    # print line.rstrip()
                    array.append([float(x) for x in line.split(",")])
            F.close
        # print array,"\n===========================================\n"
        self.layers = [int(i) for i in array[0]]
        self.__init__(self.layers)
        print self.layers
        self.w = []
        i = 0
        for layer in array[1:] :
            self.w.append(np.asarray(layer).reshape(self.layers[i+1],self.layers[i]))
            i += 1
        for layer in self.w:
            print layer

        # for a in array:
        #     print a

#methode transformant lieairement les donnees d'un tableau dans [0,1] si les valeurs sont positives, sinon dans [-1,1]
def normalize(data):
    return data/(max(np.reshape(data,data.size)))
