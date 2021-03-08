import spectrogram
from numpy import transpose
import matplotlib.pyplot as plt


# Test de la fonction inverse avec Fils du Voisin dans le but de le comparer avec le son original
[fft_mat_output, frequence_enr ] = spectrogram.spectrogramme_wav('audio/filsDuVoisin.wav', displayStretch=1, time_step=.0001, play=False, stereo=False)

#Son original
son_original,_ = spectrogram.getSound('audio/filsDuVoisin.wav')
plt.plot(son_original)
plt.title('Son original Fils du Voisin')
plt.legend()
plt.savefig('plots/Son_original_temporel.png')
plt.show()

#Son recomposé depuis le spectogramme
son = spectrogram.reconstitution_son(fft_mat_output,frequence_enr,plot=True,play=False)
matrix,T = spectrogram.matrixComputing(son,16,1,stereo=False)
spectrogram.spectroPlotting(matrix,T,1,False,'Blues')