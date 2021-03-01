import soundfile
import sounddevice as sd
from numpy.fft import fft
from numpy import abs,transpose,array,linspace,round,flip,shape,reshape
import matplotlib.pyplot as plt
from pydub import AudioSegment 
import IPython


def spectrogramme_wav(file,time_step=0.05,play=False,frequence_enregistrement=None, displayStretch = 10,cmap='Blues',stereo = False):
    #display stretch = coefficient de division de la hauteur max (frequence du graphique)
    plt.rcParams["figure.figsize"] = (17,10)
    if not stereo :
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export('temp.wav', format='wav')

        son_array,frequence_enr = soundfile.read('temp.wav')
    else :
        son_array,frequence_enr = soundfile.read(file)
    
    if play:
        sd.play(son_array,frequence_enr)

    if frequence_enregistrement is not None :
        frequence_enr = frequence_enregistrement

    N = len(son_array)
    Te = 1/frequence_enr

    T = Te * N

    step_index = int(time_step * frequence_enr)

    #Computing Matrix (matrice sautante)
    fft_mat = []
    for i in range (int(N/step_index)) :
        current_fft = abs(fft( son_array[int(i*step_index) : int((i+1)*step_index)] ))[0:int(step_index/2)] #on ne garde que la moitie des valeurs (repetition)
        fft_mat.append(current_fft)

    if stereo :
        s = shape(fft_mat)
        fft_mat= reshape(fft_mat,(s[-1],s[0],s[1]))
            
    #Plotting
   # tmp = transpose (fft_mat)
    if not stereo:
        cm = axs.imshow(transpose(fft_mat),cmap = cmap)
        fig,axs = plt.subplots()
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
    else : 
        fig,axis = plt.subplots(2,1)
        for i,axs in enumerate(axis) :
            cm  = axs.imshow(transpose(fft_mat[i]),cmap=cmap)
            ymax = axs.get_ylim()[0]
            axs.set_ylim(0,ymax/displayStretch) #empiric for size
            axs.set_aspect('auto') #Pixels arn't squares anymore
            xmax,xmin = axs.get_xlim()
            axs.set_xticks(linspace(xmin,xmax,10))
            axs.set_xticklabels(flip(linspace(0,T,10).round(2)))
            axs.set_xlabel('Temps [s]')
            axs.set_ylabel('Fréquence [Hz]') #je suis pas certain de ca 
            ymin,ymax = axs.get_ylim()
            axs.set_yticks(linspace(ymin,ymax,10))
            axs.set_yticklabels((T*linspace(ymin,ymax,10)).round(2))
            cbar = fig.colorbar(cm,ax=axs)
            cbar.set_label('Amplitude [1]')

    plt.show()

spectrogramme_wav('test.wav', displayStretch=20,time_step=0.05,play=True,stereo=True)


IPython.embed()