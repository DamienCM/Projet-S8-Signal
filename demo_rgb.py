from spectrogram import *

# ----------- INCRUSTER L'IMAGE ---------------
#  definition du pas de temps et chargement de notre son
time_step = 0.02
son_original, freq = get_sound('audio/filsDuVoisin.wav', play=False)

# On calcule la fft que l'on divise en 3 parties (avec l'argument color_sep)
fft_sans_image = matrix_computing(son_original, freq, time_step, color_sep=True)

# On ajoute de manière decomposée notre image sur chaque partie du signal
ffts_avec_image = addition_image_fft_colored('images/gros_poulet.png', fft_sans_image)

# On recompose nos FFT pour en former une seule et unique
fft_avec_image, image_originale = re_assemblage_rgb(ffts_avec_image)

# On recompose notre signal sous forme de son a partir de la FFT avec l'image
son_recomp = reconstitution_son(fft_avec_image, freq, play=True)

save_path = 'audio/son_qui_dechire_rgb.wav'

# On sauvegarde le nouveau son
save_file(son_recomp, freq, save_path)
# --------------------------------------------


# ------------- VERIFICATION ------------------

# On récupère le son enregistré sur le disque
son_verif, freq = get_sound(save_path)

#  On calcule la FFT du son dans lequel on a caché l'image
ffts_verif= matrix_computing(son_verif, freq, time_step, color_sep=True)

#  On en extrait l'image
fft_verif, image_verif = re_assemblage_rgb(ffts_verif)

# On plot et affiche tout ca
spectro_plotting(fft_avec_image, title='Image recomposee')
spectro_plotting(fft_verif, title="Image Originale")
plt.imshow(np.array(image_verif))
plt.title("Image recomposee")
plt.show()
plt.imshow(np.array(image_originale))
plt.title("Image originale")
plt.show()
# ----------------------------------------
