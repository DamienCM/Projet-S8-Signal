import spectrogram
from numpy import transpose
import matplotlib.pyplot as plt

""" TODO LIST

TODO 1 : en general sur le projet il faudrait re-verifier si tout ce que l'on a fonctionnne avec le stereo
TODO 2 : Il faut maintenant voir pour ajouter une image au son cf ci dessous

      matrixComputing 				reconstitution_son
Musique  -->  FFT_Musique  +   Image     --> 	son truqué --> Sauvegarde (Fonction a faire)
reels           complexes        reels				reels  --> Spectrogramme et verifier que l'on voit bien le son et notre image

problematique quels sont les meilleurs moyens de cacher notre image ?
-quelles frequences 
-quel temps
-quel intensite
etc

TODO 3 : en general sur le projet il faudrait set un venv pour rendre c'est bien plus propres (je fais ca soon)
"""

#Test simple comparaison entre le spectro glissant et le sautant
time_step = 0.01
son_original, freq = spectrogram.getSound('audio/filsDuVoisin.wav')
fft_mat_sautante,T_saut = spectrogram.matrixComputing(son_original,freq,time_step,False)
fft_mat_glissante,T_gliss = spectrogram.matrixComputing(son_original,freq,time_step,False,ponderation=True)
spectrogram.spectroPlotting(fft_mat_sautante, T_saut, 10, False, "Blues")
spectrogram.spectroPlotting(fft_mat_glissante, T_gliss, 10, False, "Blues")