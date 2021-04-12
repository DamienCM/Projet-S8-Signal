import numpy as np
from numpy import transpose, rot90
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import IPython
import spectrogram
import sounddevice as sd



def convertGrayscale(image): #convertit une image en nuance de gris
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


def doublement_plus1(img):
    ret = np.concatenate((np.flip(img,axis=0),np.array([np.zeros_like(img[0])])),axis=0)
    ret = np.concatenate((ret,img),axis=0)
    return ret

doublement_simple = lambda img : np.concatenate((np.flip(img,axis=0),img),axis=0)

#traitement de signal sur l'image
if __name__ == "__main__" :
    img = mpimg.imread('images/capture.png')
    #conversion de l'image
    plt.imshow(img)
    plt.savefig('./images/original.jpg')
    plt.show()
    img = convertGrayscale(img)
    # On change l'image en la doublant 
    img = doublement_plus1(img)

    plt.imshow(img,cmap='Blues')
    plt.colorbar()
    plt.savefig('./images/final.jpg')
    plt.show()
    img = np.rot90(img)
    fig,axs = spectrogram.spectroPlotting(img,519,displayStretch=1,stereo=False,cmap='Blues') #image de base dans le spectrogramme 
    son = spectrogram.reconstitution_son(img,1)

    matrix,T = spectrogram.matrixComputing(son,519,2,stereo=False,ponderation=False)
    fig,axs = spectrogram.spectroPlotting(matrix,T,1,False,'Blues') #image reconstituée dans le spectrogramme
    IPython.embed()