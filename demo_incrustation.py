from spectrogram import *

time_step = 0.01
# On charge notre son
son_original, freq = get_sound('audio/filsDuVoisin.wav', play=True)

# Calcul des matrices sautantes
fft_mat_sautante = matrix_computing_sautant(son_original, freq, time_step, False)
spectro_plotting(fft_mat_sautante, freq, title="Spectrogramme du son original", cmap="Blues")

#  On ajoute notre image à la fft
somme = addition_image_fft('images/gros_poulet.png', fft_mat_sautante, x_scale=.5, x_shift=-1, y_scale=.5,
                           y_shift=0, amplitude=1)
spectro_plotting(somme, freq, title="Spectrogramme de la fft pipee")
#  On recompose notre son a partir de la FFT sur laquelle on a ajouté le son
son_recomp = reconstitution_son(somme, freq, play=False)
fft_test = matrix_computing_sautant(son_recomp, freq, time_step, False)
spectro_plotting(fft_test, freq,title="Spectrogramme test")

# On sauvegarde notre son
output_file = 'audio/son_qui_dechire.wav'
save_file(son_recomp, freq, output_file)

#  On vérifie que l'image est bien incrustée
son_verif, freq = get_sound(output_file, play=True)
fft_verif = matrix_computing_sautant(son_verif, freq, time_step, False)
spectro_plotting(fft_verif, freq,title="Spectrogramme du son avec l'image incrustée")