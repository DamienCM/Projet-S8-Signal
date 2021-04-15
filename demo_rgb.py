from spectrogram import *

#  definition du pas de temps et chargement de notre son
time_step = 0.02
son_original, freq = get_sound('audio/filsDuVoisin.wav', play=False)

# On calcule la fft que l'on divise en 3 parties (avec l'argument color_sep)
fft_sans_image, _ = matrix_computing(son_original, freq, time_step, color_sep=True)

# On ajoute de manière decomposée notre image sur chaque partie du signal
ffts_avec_image = addition_image_fft_colored('images/gros_poulet.png', fft_sans_image)

# On recompose nos FFT pour en former une seule et unique
fft_avec_image, image_originale = re_assemblage_rgb(ffts_avec_image)

# On recompose notre so sous forme de son a partir de la FFT
son_recomp = reconstitution_son(fft_avec_image, freq, play=True)

#  On calcule la FFT du son dans lequel on a caché l'image
ffts_recomp, _ = matrix_computing(son_recomp, freq, time_step, color_sep=True)

#  On en extrait l'image
fft_recomp, image_recomp = re_assemblage_rgb(ffts_recomp)

# On plot et affiche tout ca
spectro_plotting(fft_avec_image,title='Image recomposee')
spectro_plotting(fft_recomp,title="Image Originale")
plt.imshow(np.array(image_recomp))
plt.title("Image recomposee")
plt.show()
plt.imshow(np.array(image_originale))
plt.title("Image originale")
plt.show()