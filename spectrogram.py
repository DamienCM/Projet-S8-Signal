import soundfile
import sounddevice as sd
from numpy.fft import fft, ifft
from numpy import abs, transpose, array, linspace, round, flip, shape, reshape, argmax, real, asarray, concatenate
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import IPython

def getSound(file, play=False, frequence_enregistrement=None, stereo=False):
    """
    Fonction de récupération du son
    Entrées:
    - l'adresse du fichier audio
    - un booléen définissant si le son est joué
    - la fréquence d'enregistrement
    - un booléen définissant si le son est en mono ou stéréo
    Sorties:
    - un vecteur contenant le son
    - la fréquence d'enregistrement
    """
    if not stereo:
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export('./audio/temp.wav', format='wav')

        son_array, frequence_enr = soundfile.read('./audio/temp.wav')
    else:
        son_array, frequence_enr = soundfile.read(file)

    if play:
        sd.play(son_array, frequence_enr)

    if frequence_enregistrement is not None:
        frequence_enr = frequence_enregistrement

    return son_array, frequence_enr


def matrixComputing(son_array, frequence_enr, time_step, stereo, plot=False):
    N = len(son_array)
    Te = 1 / frequence_enr
    T = Te * N

    step_index = int(time_step * frequence_enr)

    # Plotting the signal
    if plot:
        plt.plot(linspace(0, 1, len(son_array)), son_array)
        plt.title("Son au cours du temps (original)")
        plt.show()

    # Computing Matrix (matrice sautante)
    fft_mat = []
    for i in range(int(N / step_index)):
        current_fft = abs(fft(son_array[int(i * step_index): int((i + 1) * step_index)]))[
                      0:int(step_index)]  
        fft_mat.append(current_fft)

    if stereo:
        s = shape(fft_mat)
        fft_mat = reshape(fft_mat, (s[-1], s[0], s[1]))

    return fft_mat, T

def spectroPlotting(fft_mat, T, displayStretch, stereo, cmap):
    # Plotting le spectrogramme
    if not stereo:
        fig, axs = plt.subplots()
        tr =transpose(fft_mat)
        cm = axs.imshow(tr, cmap=cmap)
        ymax = axs.get_ylim()[0]
        axs.set_ylim(0, ymax / displayStretch)  # empiric for size
        axs.set_aspect('auto')  # Pixels arn't squares anymore
        xmax, xmin = axs.get_xlim()
        axs.set_xticks(linspace(xmin, xmax, 10))
        axs.set_xticklabels(flip(linspace(0, T, 10).round(2)))
        cbar = fig.colorbar(cm)
        cbar.set_label('Amplitude [1]')
        axs.set_xlabel('Temps [s]')
        axs.set_ylabel('Fréquence [Hz]')  # je suis pas certain de ca
        ymin, ymax = axs.get_ylim()
        axs.set_yticks(linspace(ymin, ymax, 10))
        axs.set_yticklabels((T * linspace(ymin, ymax, 10)).round(2))
    else:
        fig, axis = plt.subplots(2, 1)
        for i, axs in enumerate(axis):
            cm = axs.imshow(transpose(fft_mat[i]), cmap=cmap)
            ymax = axs.get_ylim()[0]
            axs.set_ylim(0, ymax / displayStretch)  # empiric for size
            axs.set_aspect('auto')  # Pixels arn't squares anymore
            xmax, xmin = axs.get_xlim()
            axs.set_xticks(linspace(xmin, xmax, 10))
            axs.set_xticklabels(flip(linspace(0, T, 10).round(2)))
            axs.set_xlabel('Temps [s]')
            axs.set_ylabel('Fréquence [Hz]')  # je suis pas certain de ca
            ymin, ymax = axs.get_ylim()
            axs.set_yticks(linspace(ymin, ymax, 10))
            axs.set_yticklabels((T * linspace(ymin, ymax, 10)).round(2))
            cbar = fig.colorbar(cm, ax=axs)
            cbar.set_label('Amplitude [1]')
    plt.title('spectrogramme du son')
    plt.show()
    return fig,axs


def spectrogramme_wav(file, time_step=0.05, play=False, frequence_enregistrement=None, displayStretch=1, cmap='Blues',
                      stereo=False):
    #display stretch = coefficient de division de la hauteur max (frequence du graphique)
    plt.rcParams["figure.figsize"] = (17, 10)

    son_array, frequence_enr = getSound(file, play, frequence_enregistrement, stereo)

    fft_mat, T = matrixComputing(son_array, frequence_enr, time_step, stereo)
   
    spectroPlotting(fft_mat,T, displayStretch, stereo, cmap)

    return [fft_mat, frequence_enr ]


def reconstitution_son(fft_mat_output, frequence_enr, play=False, plot=False):
    """
    Entrée: Matrice des fft sautantes:
    Sorties: Vecteur unidimensionnel
    """
    reconstitution = []
    asarray(reconstitution)

    for i in range(len(fft_mat_output)):
        reconstitution = concatenate([reconstitution, real(ifft(fft_mat_output[i]))])


    if play:
        sd.play(array(reconstitution), frequence_enr)

    if plot:
        plt.title('Son reconstitué')
        plt.legend()
        plt.plot(reconstitution)
        plt.savefig('plots/son_reconstitue.png')
        plt.show()

    return reconstitution





