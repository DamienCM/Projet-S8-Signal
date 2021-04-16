from spectrogram import *

time_step = 0.01
# On charge notre son
son_original, freq = get_sound('audio/filsDuVoisin.wav', play=False)

# Calcul des matrices glissantes
fft_mat_glissante = np.array(matrix_computing_glissant(son_original, freq, time_step, ponderation=True))
spectro_plotting(fft_mat_glissante, freq, title="Spectrogramme glissant", cmap="Blues")

# Calcul des matrices sautantes
fft_mat_sautant = np.array(matrix_computing_sautant(son_original, freq, time_step, ponderation=True))
spectro_plotting(fft_mat_sautant, freq, title="Spectrogramme sautant", cmap="Blues")
