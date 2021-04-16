import spectrogram
import time
import numpy as np

""" TODO LIST

TODO 1 : en general sur le projet il faudrait re-verifier si tout ce que l'on a fonctionnne avec le stereo
TODO 2 : Il faut maintenant voir pour ajouter une image au son cf ci dessous

      matrix_computing_sautant 				reconstitution_son
Musique  -->  FFT_Musique  +   Image     --> 	son truqué --> Sauvegarde (Fonction a faire)
reels           complexes        reels				reels  --> Spectrogramme et verifier que l'on voit bien le son et notre image

problématique quels sont les meilleurs moyens de cacher notre image ?
-quelles frequences 
-quel temps
-quel intensité
etc

TODO 3 : en general sur le projet il faudrait set un venv pour rendre c'est bien plus propres (je fais ca soon)
"""

# Test simple comparaison entre le spectro glissant et le sautant
time_step = 0.01
son_original, freq = spectrogram.get_sound('audio/filsDuVoisin.wav')
fft_mat_sautante, T_saut = spectrogram.matrix_computing_sautant(son_original, freq, time_step, False)
reconstitution = spectrogram.reconstitution_son(fft_mat_sautante, freq, play=True)
new_fft_mat_sautante, net_T_saut = spectrogram.matrix_computing_sautant(reconstitution, freq, time_step, False)
# spectrogram.spectro_plotting()
# time.sleep(5)
fft_mat_glissante,T_gliss = spectrogram.matrix_computing_sautant(son_original, freq, time_step, False, ponderation=True)
spectrogram.spectro_plotting(fft_mat_sautante, T_saut, 10, False, "Blues")
spectrogram.spectro_plotting(fft_mat_glissante, T_gliss, 10, False, "Blues")
