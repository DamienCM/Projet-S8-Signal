import soundfile
import sounddevice as sd
from numpy.fft import fft
from numpy import abs,transpose,array,linspace,round,flip
import matplotlib.pyplot as plt
import IPython


def spectrogramme(file, time_step=0.05, play=False,frequence_enregistrement=None, displayStretch = 10):
    #display stretch = coefficient de division de la hauteur max (frequence du graphique)
    son_array,frequence_enr = soundfile.read(file)
    if play:
        sd.play(son_array,frequence_enr)

    if frequence_enregistrement is not None :
        frequence_enr = frequence_enregistrement

    N = len(son_array)
    Te = 1/frequence_enr

    T = Te * N

    step_index = time_step * frequence_enr

    #Computing Matrix (matrice sautante)
    fft_mat = []
    for i in range (int(N/step_index)) :
        current_fft = abs(fft( son_array[int(i*step_index) : int((i+1)*step_index)] ))[0:int(step_index/2)] #on ne garde que la moitie des valeurs (repetition)
        fft_mat.append(current_fft)

    #Plotting
    fig,axs = plt.subplots()
    cm = axs.imshow(transpose(fft_mat),cmap = 'Blues')
    ymax = axs.get_ylim()[0]
    axs.set_ylim(0,ymax/displayStretch) #empiric for size
    axs.set_aspect('auto') #Pixels arn't squares anymore
    xmax,xmin = axs.get_xlim()
    axs.set_xticks(linspace(xmin,xmax,10))
    axs.set_xticklabels(flip(linspace(0,T,10).round(2)))
    cbar = fig.colorbar(cm)
    cbar.set_label('Amplitude [1]')
    axs.set_xlabel('Temps [s]')
    axs.set_ylabel('Fréquence [Hz]') #je suis pas certain de ca 
    ymin,ymax = axs.get_ylim()
    axs.set_yticks(linspace(ymin,ymax,10))
    axs.set_yticklabels((T*linspace(ymin,ymax,10)).round(2))
    plt.show()

spectrogramme('faded.wav', displayStretch=20)


IPython.embed()