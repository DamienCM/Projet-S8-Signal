import soundfile
import sounddevice as sd
import numpy as np
from PIL import Image, ImageOps
from numpy.fft import fft, ifft
from numpy import abs, transpose, array, linspace, flip, shape, reshape, real, asarray, concatenate, exp, sum, arange
import matplotlib.pyplot as plt
from pydub import AudioSegment
import time


def get_sound(file, play=False, frequence_enregistrement=None, stereo=False):
    """
    Fonction de récupération du son

    Entrées:
    - file : string de l'adresse du fichier audio 
    - play=False: booléen définissant si le son est joué
    - frequence_enregistrement=None : fréquence d'enregistrement (a preciser si l'on ne veut pas utiliser celle de base)
    - stereo=False : booléen définissant si le son est en mono ou stéréo (de base en Mono)

    Sorties:
    - son_array : un vecteur contenant le son
    - frequence_enr : la fréquence d'enregistrement
    """
    if not stereo:  # gestion mono/stereo
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export('./audio/temp.wav', format='wav')

        son_array, frequence_enr = soundfile.read('./audio/temp.wav')
    else:
        son_array, frequence_enr = soundfile.read(file)

    if play:  # jouer le son
        sd.play(son_array, frequence_enr)

    if frequence_enregistrement is not None:  # détermination de la fréquence d'enregistrement
        frequence_enr = frequence_enregistrement

    return son_array, frequence_enr


def matrix_computing(son_array, frequence_enr, time_step, stereo, plot=False, ponderation=False):
    """
    Fonction permettant de calculer la matrice des fft successives de notre signal
    
    Entrees :
    -son_array : le son sous forme d'array
    -frequence_enr : la frequence d'enregistrement du signal
    -time_step : la periode de temps sur laquelle on effectue les fft
    -plot (facultatif) boolean precisant si l'on veut plot le signal orignal
    -ponderation (falcutatif de base a False) permet de preciser si l'on veut ponderer le signal par une gaussienne

    Sorties :
    -fft_mat la matrice des fft successives
    -T le temps total du son

    TODO : Faire de meme avec le stereo et verifier
    """

    def gaussian(t, N, sigma=4):
        """
        fonction renvoyant la gausienne de ponderation
        entrees :
        t : temps array
        N : temps max 
        sigma=4 : ecart type (ou simili)
        
        retour arrray de gaussienne de ponderation normalisee
        """
        ret = exp(-(t - N / 2) ** 2 / (N / sigma) ** 2)
        return ret / sum(ret)

    N = len(son_array)
    Te = 1 / frequence_enr
    T = Te * N

    step_index = int(time_step * frequence_enr)  # equivalent d'index (entier) du time_step (secondes)

    # Plotting the signal
    if plot:
        plt.plot(linspace(0, 1, len(son_array)), son_array)
        plt.title("Son au cours du temps (original)")
        plt.show()

    # Computing Matrix
    fft_mat = []  # tableau qui va contenir l'ensemble des ffts
    for i in range(int(N / step_index)):

        if ponderation:  # Avec ponderation par la gaussienne
            current_son = son_array[int(i * step_index): int((i + 1) * step_index)]
            N = len(current_son)
            g = gaussian(arange(1, N + 1, 1), N)
            current_son = current_son * g
            current_fft = fft(current_son)

        else:  # Sans ponderation par la gaussienne
            current_fft = fft(son_array[int(i * step_index): int((i + 1) * step_index)])[
                          0:int(step_index)]

        fft_mat.append(current_fft)  # ajout de la nouvelle matrice au tableau actuel

    if stereo:  # Remise en forme des données dans le cas d'un son stéréo
        s = shape(fft_mat)
        fft_mat = reshape(fft_mat, (s[-1], s[0], s[1]))

    return fft_mat, T


def spectro_plotting(fft_mat, T, displayStretch=1, title ="Spectrogramme", stereo=False, cmap='Blues'):
    """
    Fonction d'affichage de notre spectrogramme.

    Entrees :
    -fft_mat : array des fft successives de notre son
    -T : temps total de l'enregistrement
    -displaystretch=1  argument permettant de regler la hauteur de l'affichage (augmenter pour zommer sur y)
    -stereo permet de preciser si notre signal est en stereo ou non
    -cmap='Blues' : color map matplolib utilisee pour la representation, parce que le Blues c'est cool comme musique

    Sortie 
    fig : figure matplotlib
    axs : ax matplotlib
    """
    # Plotting le spectrogramme
    fft_mat = abs(fft_mat)
    if not stereo:
        # creation de la figure
        fig, axis = plt.subplots()
        # On transpose la matrice pour l'afficher dans le bon sens
        tr = transpose(fft_mat)
        # affichage et reglages des limites
        cm = axis.imshow(tr, cmap=cmap)
        ymax = axis.get_ylim()[0]
        axis.set_ylim(0, ymax / displayStretch)
        axis.set_aspect('auto')  # Les pixels ne sont plus carrés
        xmax, xmin = axis.get_xlim()
        axis.set_xticks(linspace(xmin, xmax, 10))
        axis.set_xticklabels(flip(linspace(0, T, 10).round(2)))
        cbar = fig.colorbar(cm)
        cbar.set_label('Amplitude [1]')
        axis.set_xlabel('Temps [s]')
        axis.set_ylabel('Fréquence [Hz]')
        ymin, ymax = axis.get_ylim()
        axis.set_yticks(linspace(ymin, ymax, 10))
        axis.set_yticklabels((T * linspace(ymin, ymax, 10)).round(2))
    else:
        fig, axis = plt.subplots(2, 1)
        for i, axs in enumerate(axis):
            cm = axs.imshow(transpose(fft_mat[i]), cmap=cmap)
            ymax = axs.get_ylim()[0]
            axs.set_ylim(0, ymax / displayStretch)
            axs.set_aspect('auto')  # Les pixels ne sont plus carré
            xmax, xmin = axs.get_xlim()
            axs.set_xticks(linspace(xmin, xmax, 10))
            axs.set_xticklabels(flip(linspace(0, T, 10).round(2)))
            axs.set_xlabel('Temps [s]')
            axs.set_ylabel('Fréquence [Hz]')
            ymin, ymax = axs.get_ylim()
            axs.set_yticks(linspace(ymin, ymax, 10))
            axs.set_yticklabels((T * linspace(ymin, ymax, 10)).round(2))
            cbar = fig.colorbar(cm, ax=axs)
            cbar.set_label('Amplitude [1]')
    plt.title(title)
    plt.show()
    return fig, axis


def spectrogramme_wav(file, time_step=0.05, play=False, frequence_enregistrement=None, displayStretch=1, cmap='Blues',
                      stereo=False):
    """
    Fonction all in one pour dessiner le spectrogramme a partir d'un fichier son
    
    Entrees :
    -file : string path vers le fichier son
    -time_step=0.05 : pas de temps avec lequel on effectue les ffts
    -play=False : permet de jouer le son si True
    -frequence_enregistrement=None : permet de preciser une frequence pour l'enregistrement si ce n'est pas celle de base
    -displayStrech=1 : permet de compresser l'affichage sur y (ie zoomer)
    -cmap="Blues" : colormap utilisee pour le spectrogramme
    
    Sorties :
    -fft_mat : la matrices des fft successives
    -frequence_enregistrement : la frequence a utiliser avec la fft pour reconstituer le son
    """
    # display stretch = coefficient de division de la hauteur max (frequence du graphique)
    plt.rcParams["figure.figsize"] = (17, 10)

    # Récupération du son
    son_array, frequence_enr = get_sound(file, play, frequence_enregistrement, stereo)

    # Calcul de la matrice des FFT
    fft_mat, T = matrix_computing(son_array, frequence_enr, time_step, stereo)

    # Dessin du spectrogramme
    spectro_plotting(fft_mat, T, displayStretch, stereo, cmap)

    return fft_mat, frequence_enr


def reconstitution_son(fft_mat_output, frequence_enr, play=False, plot=False):
    """
    Fonction permettant de reconstituer un son a partir d'une matrice interpretee comme une matrice de fft successives.

    Entrée: 
    -fft_mat_output : Matrice des fft sautantes (construite via matrix_computing)
    -frequence_enr : frequence de l'enregistrement audio a reconstruire
    -play=False : Boolean definissant si l'on veut jouer le son reconstruit ou non 
    -plot=False : Boolean definissant si l'on veut plot temporellement le son reconstruit

    Sorties: 
    -Son reconstitué (vecteur unidimentionnel)

    TODO : Faire le stereo
    """
    # On crée une liste vide transformée en vecteur numpy
    reconstitution = []
    asarray(reconstitution)

    # On applique la FFT inverse sur l'ensemble des pas de temps
    for i in range(len(fft_mat_output)):
        reconstitution = concatenate([reconstitution, real(ifft(fft_mat_output[i]))])  # Concaténation des FFT inverses

    # On joue le son reconstitué
    if play:
        sd.play(array(reconstitution), frequence_enr)

    # Dessin du son reconstitué
    if plot:
        plt.title('Son reconstitué')
        plt.legend()
        plt.plot(reconstitution)
        plt.savefig('plots/son_reconstitue.png')
        plt.show()

    return reconstitution


def addition_image_son(image_path, fft_son, x_shift=0., y_shift=0., x_size=.7, y_size=1.):
    """
    Fonction permettant d'ajouter l'image à la matrice des fft du son
    
    Entree :
    -image_path : string du chemin vers l'image
    -fft_son : ndarray complexe de la fft notre son
    -x_shift : coefficient floatant de decalage sur x de l'emplacement ou l'on va ajouter l'image dans la fft (-1; 1) --> 0 = centre
    -y_shift : coefficient floatant de decalage sur y de l'emplacement ou l'on va ajouter l'image dans la fft (-1;1) --> 0 = centre
    -x_size : place que prendra l'image sur x dans la fft (de base pleine echelle)
    -y_size : place que prendra l'image sur y dans la fft (de base pleine echelle)

    Sorties :
    -fft_ret : nouvelle fft que l'on renvoie
    """
    # On transpose la matrice fft pour travailler en superposition avec le spectro
    fft_tr = transpose(fft_son)  # on laisse abs pour le moment pour tester avec des reels

    m, n = np.shape(fft_tr)  # dimensions de l'array fft_tr qui nous servent reshape notre image
    # charge l'image
    img = Image.open(image_path)
    # On resize l'image pour l'adapter à la fft et on la double
    img = img.resize((int(n*x_size), int(m / 2)), resample=Image.BOX)
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img_mat = np.array(img)
    h, l = np.shape(img_mat)  # dimensions de la matrice reconvertie
    # test des conditions de validités
    gauche = int(n / 2 + x_shift*n/2 - l / 2)
    droite = n - l - gauche
    if gauche < 0:
        raise ValueError("Erreur : x_shift petit,  l'image est trop à gauche")
    if droite < 0:
        raise ValueError("Erreur : x_shift grand,  l'image est trop à droite")
    img_mat = doublement(img_mat)
    droite_mat = np.zeros((m, droite))
    gauche_mat = np.zeros((m, gauche))
    img_mat = np.concatenate((gauche_mat, img_mat, droite_mat),axis=1)
    img_mat = img_mat * np.max(fft_tr) / np.max(img_mat)
    somme = img_mat + fft_tr
    fft_ret = transpose(somme)
    return fft_ret


def doublement(img, simple=True):
    """
    Fonction permettant de doubler l'image grayscale en hauteur

    Entree :
    -image : array contenant l'image N x M x 1
    -simple=True : Boolean précisant si l'on veut doubler simplement l'image (True) ou si l'on veut la doubler en inserant un pixel blanc au milieu (False)

    Sortie :
    -ret : image doublée 2N x M x 1
    """
    if simple:
        return np.concatenate((np.flip(img, axis=0), img), axis=0)
    else:
        ret = np.concatenate((np.flip(img, axis=0), np.array([np.zeros_like(img[0])])), axis=0)
        ret = np.concatenate((ret, img), axis=0)
    return ret


def save_file(son_array, frequence):
    return


if __name__ == '__main__':
    plt.interactive(False)
    time_step = 0.01
    #  son original
    son_original, freq = get_sound('audio/filsDuVoisin.wav', play=False)
    fft_mat_sautante, T_saut = matrix_computing(son_original, freq, time_step, False)
    spectro_plotting(fft_mat_sautante, T_saut, 1, title="Spectrogramme du son original" , cmap="Blues")

    #  addition
    somme = addition_image_son('images/JBH.png', fft_mat_sautante,x_size=.2, x_shift=.9)
    spectro_plotting(somme, T_saut, 1, title="Spectrogramme du son original additionne avec l'image", cmap="Blues")

    #  son recompose
    son_recomp = reconstitution_son(somme, freq, play=True)
    fft_mat_sautante, T_saut = matrix_computing(son_recomp, freq, time_step, False)
    spectro_plotting(fft_mat_sautante, T_saut, 1, title="Spectrogramme du son recompose", cmap="Blues")
    time.sleep(5)
