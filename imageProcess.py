import numpy as np
from numpy import transpose, rot90
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import IPython
import spectrogram
import sounddevice as sd



def convertGrayscale(image): 
    """
    Convertit une image en nuance de gris 

    Entrees :
    -image : array N x M x 3 contentant l'image RGB

    Sorties 
    -image : array N x M x 1 contentant l'image en Grayscales

    TODO : Ameliorer en utilisant numpy parce la c'est tres long avec les listes
    """
    tmp = []
    for i in range(len(image)):
        tmp1 =[]
        for j in range(len(image[i])):
            if image[i][j][3] == 0 : #si la case est transparente elle devient blanche
                tmp1.append(0)
            elif  image[i][j][3] == 1 and image[i][j][0] == 0 and image[i][j][1] == 0 and image[i][j][2] == 0: # si la case est blanche et completement opaque  ie noire [0,0,0,1] elle devient noire
                    tmp1.append(1)           
            else:
                tmp1.append( 0.2126*image[i][j][0] + 0.7152*image[i][j][1]+0.0722*image[i][j][2])
            #0,2126 × Rouge + 0,7152 × Vert + 0,0722 × Bleu. pour obtenir le grayscale
        tmp.append(tmp1)
    return np.array(tmp)


def doublement(img,simple=True):
    """
    Fonction permettant de doubler l'image grayscale en hauteur

    Entree : 
    -image : array contenant l'image N x M x 1
    -simple=True : Boolean precisant si l'on veut doubler simplement l'image (True) ou si l'on veut la doubler en inserant un pixel blanc au milieu (False)
    
    Sortie : 
    -ret : image doublee 2N x M x 1
    """
    if simple :
        return np.concatenate((np.flip(img,axis=0),img),axis=0)
    else:
        ret = np.concatenate((np.flip(img,axis=0),np.array([np.zeros_like(img[0])])),axis=0)
        ret = np.concatenate((ret,img),axis=0)
    return ret
 







#traitement de signal sur l'image
if __name__ == "__main__" :
    #Chargement de l'image
    img = mpimg.imread('images/capture.png')
    plt.imshow(img)
    plt.savefig('./images/original.jpg')
    plt.show()
    
    #Convertit l'image en Grayscale puis on la double et l'affiche
    img = convertGrayscale(img)
    img = doublement(img)
    plt.imshow(img,cmap='Blues')
    plt.colorbar()
    plt.savefig('./images/final.jpg')
    plt.show()
    
    #On tourne l'image de 90 deg (parce qu'elle est transposee dans le spectro)
    img = np.rot90(img)
    #On la plot en tant que spectrogramme
    fig,axs = spectrogram.spectroPlotting(img,519,displayStretch=1,stereo=False,cmap='Blues') #image de base dans le spectrogramme 
    
    #On reconstitue un son a partir de l'inage
    son = spectrogram.reconstitution_son(img,1)

    #On calcule les FFT de ce son issu de l'image
    matrix,T = spectrogram.matrixComputing(son,519,2,stereo=False,ponderation=False)
    
    #On plot le resultat
    fig,axs = spectrogram.spectroPlotting(matrix,T,1,False,'Blues') #image reconstituée dans le spectrogramme
    IPython.embed()