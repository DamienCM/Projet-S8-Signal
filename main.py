import spectrogram
from numpy import transpose
[fft_mat_output, frequence_enr ] = spectrogram.spectrogramme_wav('audio/filsDuVoisin.wav', displayStretch=1, time_step=0.05, play=False, stereo=False)

son = spectrogram.reconstitution_son(transpose(fft_mat_output),frequence_enr,plot=True,play=False)
son = transpose(son[0:int(len(son/2))])
matrix,T = spectrogram.matrixComputing(son,16,1,stereo=False)
spectrogram.spectroPlotting(matrix,T,1,False,'Blues')