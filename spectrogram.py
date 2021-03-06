import soundfile
import sounddevice as sd
from numpy.fft import fft, ifft
from numpy import abs, transpose, array, linspace, round, flip, shape, reshape, argmax, real, asarray, concatenate
import matplotlib.pyplot as plt
from pydub import AudioSegment
import IPython


def spectrogramme_wav(file, time_step=0.05, play=False, frequence_enregistrement=None, displayStretch=10, cmap='Blues',
                      stereo=False):
    # display stretch = coefficient de division de la hauteur max (frequence du graphique)
    plt.rcParams["figure.figsize"] = (17, 10)
    if not stereo:
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export('temp.wav', format='wav')

        son_array, frequence_enr = soundfile.read('temp.wav')
    else:
        son_array, frequence_enr = soundfile.read(file)

    if play:
        sd.play(son_array, frequence_enr)

    if frequence_enregistrement is not None:
        frequence_enr = frequence_enregistrement

    N = len(son_array)
    Te = 1 / frequence_enr

    T = Te * N

    step_index = int(time_step * frequence_enr)

    # Plotting the signal
    plt.plot(linspace(0, 1, len(son_array)), son_array)
    plt.show()

    # Computing Matrix (matrice sautante)
    fft_mat = []
    for i in range(int(N / step_index)):
        current_fft = abs(fft(son_array[int(i * step_index): int((i + 1) * step_index)]))[
                      0:int(step_index / 2)]  # on ne garde que la moitie des valeurs (repetition)
        fft_mat.append(current_fft)
        # print(shape(current_fft))
        if(i == 101):
            # print(current_fft)
            # print(shape(current_fft))
            print(abs(ifft(current_fft)))
            reversed_sound = real(ifft(current_fft))
            plt.plot(linspace(0, 1, len(reversed_sound)), reversed_sound)
            plt.title("test")
            plt.show()

    # print(shape(fft_mat))

    if stereo:
        s = shape(fft_mat)
        fft_mat = reshape(fft_mat, (s[-1], s[0], s[1]))

    # Plotting
    # tmp = transpose (fft_mat)
    if not stereo:
        fig, axs = plt.subplots()
        cm = axs.imshow(transpose(fft_mat), cmap=cmap)
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

    plt.show()

    return [fft_mat, frequence_enr ]

def ispectrogramme_wav(fft_mat_output, frequence_enr, play=True, plot=True):
    """
    Entrée: Matrice des fft sautantes:
        - lignes:
        - colonnes:
    Sorties: Vecteur unidimensionnel
    """
    reconstitution = []
    asarray(reconstitution)

    for i in range(len(fft_mat_output)):
        reconstitution = concatenate([reconstitution, real(ifft(fft_mat_output[i]))])

    if play:
        sd.play(reconstitution, frequence_enr)

    if plot:
        plt.title('Son reconstitué')
        plt.plot(linspace(0, 1, len(reconstitution)), reconstitution)
        plt.show()

    return reconstitution

[fft_mat_output, frequence_enr ] = spectrogramme_wav('LA_mono.wav', displayStretch=20, time_step=0.05, play=False, stereo=False)

ispectrogramme_wav(fft_mat_output, frequence_enr)

IPython.embed()