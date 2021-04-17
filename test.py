from spectrogram import *
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
son_original, freq = get_sound('audio/filsDuVoisin.wav', play=False)

time_step = 0.01
# On charge notre son
son_original, freq = get_sound('audio/filsDuVoisin.wav', play=False)

# Calcul des matrices sautantes
fft_mat_sautante = matrix_computing_sautant(son_original, freq, time_step, ponderation=True)
spectro_plotting(fft_mat_sautante, freq, displayStretch=20,title="", cmap="Blues")


nom = "sautant_avec"

plt.xlabel('Temps (s)')
plt.ylabel('Frequence (Hz)')

plt.savefig(f'figures/{nom}.png')
tikzplotlib.save(f'figures/{nom}.tex',axis_width='.8\linewidth',axis_height='0.4\linewidth') 
plt.show()