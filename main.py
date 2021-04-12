import spectrogram
from numpy import transpose
import matplotlib.pyplot as plt

'''
TODO :
Il faut trouver pour chaque paire de pixel dans un colomne du spectro la valeur complexe qui une fois IFFT donne un signal reel (ou presque)

    I   ------->    C   ------>   S
  Float          Complexe       Signal
  Image          FFT_MAT         Reel


Ie : Il faut reussir a trouver une etape qui permet de passer de I a C

'''
# Test de la fonction inverse avec Fils du Voisin dans le but de le comparer avec le son original
#[fft_mat_output, frequence_enr ] = spectrogram.spectrogramme_wav('audio/filsDuVoisin.wav', displayStretch=1, time_step=.0001, play=False, stereo=False)

#Son original
time_step = 0.01
son_original, freq = spectrogram.getSound('audio/filsDuVoisin.wav')
fft_mat_sautante,T_saut = spectrogram.matrixComputing(son_original,freq,time_step,False)
fft_mat_glissante,T_gliss = spectrogram.matrixComputing(son_original,freq,time_step,False,ponderation=True)
spectrogram.spectroPlotting(fft_mat_sautante, T_saut, 10, False, "Blues")
spectrogram.spectroPlotting(fft_mat_glissante, T_gliss, 10, False, "Blues")


#plt.plot(son_original)
# plt.title('Son original Fils du Voisin')
# plt.legend()
# plt.savefig('plots/Son_original_temporel.png')
# plt.show()

#Son recomposé depuis le spectogramme
# son = spectrogram.reconstitution_son(fft_mat_output,frequence_enr,plot=True,play=False)
# matrix,T = spectrogram.matrixComputing(son,16,1,stereo=False)
# spectrogram.spectroPlotting(matrix,T,1,False,'Blues')