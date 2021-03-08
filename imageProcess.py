import numpy as np
from numpy import transpose, rot90
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import IPython
import spectrogram

img = mpimg.imread('images/JBH.png')
def convertGrayscale(image):
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
            #0,2126 × Rouge + 0,7152 × Vert + 0,0722 × Bleu. 
        tmp.append(tmp1)
    return np.array(tmp)


plt.imshow(img)
plt.savefig('./images/original.jpg')
plt.show()
img = convertGrayscale(img)
plt.imshow(img,cmap='Blues')
plt.colorbar()
plt.savefig('./images/final.jpg')
plt.show()

spectrogram.spectroPlotting(rot90(img,k=-1),16,displayStretch=1,stereo=False,cmap='Blues')
son = spectrogram.reconstitution_son(img,1)
matrix,T = spectrogram.matrixComputing(son,16,1,stereo=False)
spectrogram.spectroPlotting(matrix,T,1,False,'Blues')

IPython.embed()