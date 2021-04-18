import soundfile
import sounddevice as sd
import numpy as np
from PIL import Image, ImageOps
from numpy.fft import fft, ifft
from numpy import abs, transpose, array, linspace, real, asarray, concatenate, exp, sum, arange
import matplotlib.pyplot as plt
from pydub import AudioSegment
from tqdm import tqdm


def gaussian(t, n, sigma=4):
    """
    fonction renvoyant la gausienne de ponderation pour faire des spectrogramme plus jolis
    entrees :
    t : temps array
    n : temps max
    sigma=4 : écart type (ou simili)

    retour arrray de gaussienne de ponderation normalisee
    """
    ret = exp(-(t - n / 2) ** 2 / (n / sigma) ** 2)
    return ret / sum(ret)

def bartlett(n):
    """
    fonction renvoyant la bartlett de ponderation pour faire des spectrogramme plus jolis
    entrees :
    n : temps max

    retour arrray de bartlett de ponderation normalisee
    """
    L = []
    for i in range(n):
        if i<n/2:
            L.append(2*(i)/n)
        else :
            L.append(2*(n-i)/n)
    ret = np.array(L)
    return ret / sum(ret)

def hann(n):
    """
    fonction renvoyant la hann de ponderation pour faire des spectrogramme plus jolis
    entrees :
    n : temps max

    retour arrray de hann de ponderation normalisee
    """

    ret = np.linspace(0,n,n)
    ret = np.ones_like(ret)*.5 -.5*np.cos(2*np.pi*ret/n)
    return ret / sum(ret)


def get_sound(file, play=False, frequence_enregistrement=None):
    """
    Fonction de récupération du son depuis un fichier .wav

    Entrées:
    - file : string de l'adresse du fichier audio 
    - play=False: booléen définissant si le son est joué
    - frequence_enregistrement=None : fréquence d'enregistrement (a preciser si l'on ne veut pas utiliser celle de base)

    Sorties:
    - son_array : un vecteur contenant le son
    - frequence_enr : la fréquence d'enregistrement
    """

    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    sound.export('./audio/temp.wav', format='wav')

    son_array, frequence_enr = soundfile.read('./audio/temp.wav')

    if play:  # jouer le son
        sd.play(son_array, frequence_enr)

    if frequence_enregistrement is not None:  # détermination de la fréquence d'enregistrement
        frequence_enr = frequence_enregistrement

    return son_array, frequence_enr


def matrix_computing_sautant(son_array, frequence_enr, time_step, plot=False, ponderation=False, color_sep=False, ponderation_type='gaussian'):
    """
    Fonction permettant de calculer la matrice des fft successives sautantes de notre signal
    
    Entrees :
    -son_array : le son sous forme d'array
    -frequence_enr : la fréquence d'enregistrement du signal
    -time_step : la période de temps sur laquelle on effectue les fft
    -plot=False (facultatif) boolean précisant si l'on veut plot le signal orignal (deconseillé)
    -ponderation=False (falcutatif) permet de preciser si l'on veut ponderer le signal par une gaussienne

    Sorties :
    -fft_mat la matrice des fft successives
    """
    ponderation_set = {"gaussian","bartlett",'hann'}


    N = len(son_array)
    Te = 1 / frequence_enr
    T = Te * N  # Temps total de l'enregistrement 

    step_index = int(time_step * frequence_enr)  # equivalent d'index (entier) du time_step (secondes)

    # Plotting the signal
    if plot:
        plt.plot(linspace(0, 1, len(son_array)), son_array)
        plt.title("Son au cours du temps (original)")
        plt.show()

    # Computing Matrix
    fft_mat = []  # tableau qui va contenir l'ensemble des ffts
    print("Calcul de la matrice de la FFT de la musique")
    for i in tqdm(range(int(N / step_index))):

        if ponderation:  # Avec ponderation par la gaussienne
            if ponderation_type not in ponderation_set :
                raise ValueError(f"Type de ponderation inconnue {ponderation_type}")
            current_son = son_array[int(i * step_index): int((i + 1) * step_index)]
            N = len(current_son)
            if ponderation_type =='gaussian':
                g = gaussian(arange(1, N + 1, 1), N)
            if ponderation_type == 'bartlett':
                g = bartlett(N)
            if ponderation_type == 'hann':
                g=hann(N)
            current_son = current_son * g
            current_fft = fft(current_son)

        else:  # Sans ponderation par la gaussienne
            current_fft = fft(son_array[int(i * step_index): int((i + 1) * step_index)])[
                          0:int(step_index)]

        fft_mat.append(current_fft)  # ajout de la nouvelle matrice au tableau actuel

    # On normalise les amplitudes
    fft_mat = 255 * np.array(fft_mat) / np.max(fft_mat)

    #  Dans le cas ou l'on veut travailler en RGB
    if color_sep:
        pas = int(len(fft_mat) / 3)
        fft_mat = (fft_mat[0:pas], fft_mat[pas:2 * pas], fft_mat[2 * pas:3 * pas])

    return fft_mat


def matrix_computing_glissant(son_array, frequence_enr, time_step, ponderation=False):
    """
    Fonction permettant de calculer la matrice des fft successives glissante de notre signal
    
    Entrees :
    -son_array : le son sous forme d'array
    -frequence_enr : la fréquence d'enregistrement du signal
    -time_step : la période de temps sur laquelle on effectue les fft
    -ponderation=False (falcutatif) permet de preciser si l'on veut ponderer le signal par une gaussienne

    Sorties :
    -fft_mat la matrice des fft successives
    """
    N = len(son_array)
    step_index = int(time_step * frequence_enr)  # equivalent d'index (entier) du time_step (secondes)

    # Computing Matrix
    fft_mat = []  # tableau qui va contenir l'ensemble des ffts
    print("Calcul de la matrice de la FFT glissante de la musique")
    for i in tqdm(range(int(N - step_index))):
        current_son = np.array(son_array[i: i + step_index])
        if ponderation:
            N = len(current_son)
            g = gaussian(arange(1, N + 1, 1), N)
            current_son = g * current_son
        current_fft = fft(current_son)
        fft_mat.append(current_fft)  # ajout de la nouvelle matrice au tableau actuel

    return fft_mat


def spectro_plotting(fft_mat, freq=1, displayStretch=2, title="Spectrogramme", cmap='Blues'):
    """
    Fonction d'affichage de notre spectrogramme.

    Entrees :
    -fft_mat : array des fft successives de notre son
    -T : temps total de l'enregistrement
    -displaystretch=1  argument permettant de regler la hauteur de l'affichage (augmenter pour zommer sur y)
    -cmap='Blues' : color map matplolib utilisee pour la representation, parce que le Blues c'est cool comme musique

    Sortie 
    fig : figure matplotlib
    axs : ax matplotlib
    """
    print('Plotting spectrogram : ...')
    # Plotting le spectrogramme
    fft_mat = abs(fft_mat)
    # creation de la figure
    fig, axis = plt.subplots()
    # On transpose la matrice pour l'afficher dans le bon sens
    tr = transpose(fft_mat)
    # affichage et reglages des limites
    cm = axis.imshow(tr, cmap=cmap)

    axis.set_aspect('auto')  # Les pixels ne sont plus carrés
    xmax, xmin = axis.get_xlim()
    axis.set_xticks(linspace(xmin, xmax, 6))
    axis.set_xticklabels((linspace(1 / freq * np.size(fft_mat), 0, 6)).round(1))

    ymax = axis.get_ylim()[0]
    axis.set_ylim(0, ymax / displayStretch)

    ymin, ymax = axis.get_ylim()
    axis.set_yticks(linspace(ymin, ymax, 6))
    xtl = [f"{i}k" for i in np.int32(freq / 1000 * linspace(0, 1 / displayStretch, 6))]
    axis.set_yticklabels(xtl)

    cbar = fig.colorbar(cm)
    cbar.set_label('Amplitude')
    axis.set_xlabel('Temps [s]')
    axis.set_ylabel('Fréquence [Hz]')

    plt.title(title)
    print('Plotting spectrogram : Done')
    plt.show()
    return fig, axis



def reconstitution_son(fft_mat_output, frequence_enr=44000, play=False, plot=False):
    """
    Fonction permettant de reconstituer un son a partir d'une matrice interpretee comme une matrice de fft successives.

    Entrée: 
    -fft_mat_output : Matrice des fft sautantes (construite via matrix_computing_sautant)
    -frequence_enr : frequence de l'enregistrement audio a reconstruire
    -play=False : Boolean definissant si l'on veut jouer le son reconstruit ou non 
    -plot=False : Boolean definissant si l'on veut plot temporellement le son reconstruit

    Sorties: 
    -Son reconstitué (vecteur unidimentionnel)

    """
    # On crée une liste vide transformée en vecteur numpy
    reconstitution = []
    asarray(reconstitution)
    print("Calcul du nouveau son")
    # On applique la FFT inverse sur l'ensemble des pas de temps
    for i in tqdm(range(len(fft_mat_output))):
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

    return reconstitution / max(reconstitution)  # /max(reconstitution) HYPER IMPORTANT


def addition_image_fft(image_path, fft_son, amplitude=1., x_scale=.5, x_shift=0., y_scale=.5, y_shift=0.):
    """
    Fonction permettant d'ajouter l'image à la matrice des fft du son
    
    Entree :
    -image_path : string du chemin vers l'image
    -fft_son : ndarray complexe de la fft notre son
    -x_scale : coefficient floatant de decalage sur x de l'emplacement ou l'on va ajouter l'image dans la fft (-1; 1) --> 0 = centre
    -y_shift : coefficient floatant de decalage sur y de l'emplacement ou l'on va ajouter l'image dans la fft (-1;1) --> 0 = centre
    -y_scale : place que prendra l'image sur x dans la fft (de base pleine echelle)
    -y_size : place que prendra l'image sur y dans la fft (de base pleine echelle)

    Sorties :
    -fft_ret : nouvelle fft que l'on renvoie
    """

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

    # On transpose la matrice fft pour travailler en superposition avec le spectro
    fft_tr = transpose(fft_son)

    # charge l'image et la convertit en grayscale
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)

    n, m = img.size
    img_mat = np.array(img)  # convertit l'image grayscale en array exploitable
    print("Ajout de l'image à la FFT : ...")
    # Pour x on scale et decale
    try:
        # scaling (on entoure a droite et a gauche de matrices de 0)
        bord = np.zeros((m, int(1 / x_scale * n - n)))
        img_mat = np.concatenate((bord, img_mat, bord), axis=1)

        # decalage (on concatene avec une matrice de 0)
        if x_shift > 0:
            bord = np.zeros((m, int(n * x_scale)))
            img_mat = np.concatenate((bord, img_mat), axis=1)
        if x_shift < 0:
            bord = np.zeros((m, int(n * abs(x_scale))))
            img_mat = np.concatenate((img_mat, bord), axis=1)
    except ValueError:
        # En cas d'erreur lors de l'addition, on ne la fait pas
        print("Erreur lors de x_scale ou x_shift on continue sans")

    m, n = np.shape(img_mat)

    # De meme pour y
    try:
        bord = np.zeros((int(1 / y_scale * m - m), n))
        img_mat = np.concatenate((bord, img_mat, bord), axis=0)
        if y_shift > 0:
            bord = np.zeros((int(m * y_shift), n))
            img_mat = np.concatenate((img_mat, bord), axis=0)
        if y_shift < 0:
            bord = np.zeros((int(m * abs(y_shift)), n))
            img_mat = np.concatenate((bord, img_mat), axis=0)
    except ValueError:
        print("Erreur lors de y_shift ou y_shift on continue sans")

    # On double l'image pour ne pas oublier les frequences negatives
    img_mat = doublement(img_mat)

    # On resize aux dimensions de la matrice des fft
    img = Image.fromarray(img_mat)
    n, m = np.shape(fft_tr)
    img = img.resize((m, n), resample=Image.BOX)
    img_mat = np.array(img)

    # On fait la somme emtre la matrice complexe des fft et la matrice de reels de l'image
    if amplitude > 1:
        amplitude = 1
    img_mat = img_mat * np.max(fft_tr) / np.max(img_mat) * amplitude
    somme = img_mat + fft_tr
    somme = somme * 255 / np.max(somme)

    # On retourne la matrice de somme
    fft_ret = transpose(somme)
    print("Ajout de l'image à la FFT : Done")
    return fft_ret


def addition_image_fft_colored(image, ffts):
    """
    Ajoute les 3 composantes RGB sur la FFT de notre signal (1 couleur par tiers)
    @param image: path vers l'image
    @param ffts: tuple contenant les 3 tiers du signal en fft
    @return: list des 3 tiers de la fft chacun implementé d'une partie de l'image
    """
    # Ouverture de l'image
    img = Image.open(image)
    data = img.getdata()
    img = ImageOps.grayscale(img)

    # On recupere les composantes sur r,g,b
    rgbs = [[d[0] for d in data],
            [d[1] for d in data],
            [d[2] for d in data]
            ]
    fft_ret = []
    #  Pour chaque tiers du signal
    for i, fft in enumerate(ffts):
        # On l'image par composante R,G,B
        # premier tiers : composante rouge de l'image, 2nd tiers composante verte, etc
        img.putdata(rgbs[i])
        img.save(f'images/{i}.png')  # image qui va etre injectee sur ce tiers
        fft_ret.append(addition_image_fft(f'images/{i}.png', fft))  # injection sur ce tiers
    return fft_ret


def save_file(son_array, frequence, path='audio/out.wav'):
    """
    Fonction permettant d'enregistrer notre son recomposé

    @param son_array: array de notre son
    @param frequence: frequence d'enregistrement
    @param path: chemin du fichier que l'on veut enregistrer
    """
    soundfile.write(path, son_array, frequence, format='WAV')


def re_assemblage_rgb(matrices_fft):
    """

    @param matrices_fft: les 3 fft a re-assembler
    @return: big_fft --> la fft re-assemblée
             img_reconstitue --> l'image décodée (PIL Image)
    """
    # fft complete du signal en remettant bout a bout
    big_fft = np.concatenate(matrices_fft, axis=0)

    #  décodage de l'image
    img_reconstitue = np.stack((matrices_fft), axis=-1)  # 3 matrices n x m ---> 1 matrice n x m x 3
    img_reconstitue = np.uint8(np.rot90(img_reconstitue)[:][int(len(img_reconstitue[1]) / 2):][:])  # Crop et rotate
    img_reconstitue = Image.fromarray(img_reconstitue)  # Conversion en PIL.Image et inversion des couleurs
    img_reconstitue = ImageOps.invert(img_reconstitue)
    return big_fft, img_reconstitue


if __name__ == '__main__':
    pass
