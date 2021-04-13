# Projet
 Mini Projet de Signal S8 sur la dissimulation d'images dans un son au travers de son spectrogramme
---------------------
Pour le rendu intermediaire ne pas prendre en compte le script main2.py 
---------------------

# Initialisation du projet et installation des dependances
Creation d'un environnement virtuel :
1. Lancer un terminal (powershell sur windows)
2. Se placer dans le repertoire de travail avec la commande cd (ie le repertoire Projet)
3. Lancer les commandes suivantes afin de creer un environement virtuel, d'installer  les dependances et de l'activer
    * Sur windows
    >$ python -m venv venv 

    >$ . venv/Scripts/activate.ps1 

    >$ pip install -r requirements.txt
    * Sur Linux


Le projet est constitué de deux sous dossiers et de plusieurs script et librairies : 
* audio :  
 Ce dossier contient plusieurs fichiers audio permettant de tester nos scripts

*  images  
Ce dossier permet de tester nos scripts avec de multiples images 

* spectrogram.py 

    Il s'agit de la  librairie principale du projet dans laquelle sont repertoriées la majorité des fonctions que nous utiliserons, elle est articulée autour d'une fonction principale : **spectrogram_wav()**, de plusieurs fonctions annexes qui en sont la décomposition ainsi que de **reconstitution_son**

    * spectrogramme_wav(file, time_step=0.05, play=Fals    frequence_enregistrement=None, displayStretch=10, cmap='Blues'  stereo=False)
    
        Fonction permettant d'afficher le spectrogramme d'un son **.wav** dans un graph

        Retour : 
        * fft_mat : ndarray, contenant le spectrogramme sous forme matricielle
        * frequence_enr : float/int, frequence de l'enregistrement du son

        Arguments : 
        * file : string, emplacement du fichier sonore
        * time_step : float, temps avec lequel on effectue les transformées de Fourier sautantes
        * play : boolean (optional), permet de jouer ou non le son dans les hauts parleurs
        * frequence_enregistrement : float/int (optionnal), permet de preciser la frequence de l'enregistement si on veut qu'elle differe de celle originale
        * displayStretch : float (optionnal) permet de compresser plus ou moins l'affichage vertical du graph
        * cmap : string (optionnal), permet de preciser la colormap que l'on souhaite utiliser pour le graphique
        * **stereo** : boolean, permettant de preciser si le son est stereo ou non


    * getSound(file, play=False, frequence_enregistrement=None, stereo=False)
    
        Fonction permettant d'obtenir la matrice représentative d'un fichier son

        Retour :
         * son_array : ndarray, matrice représentative du son audio
         * frequence_enr : float, fréquence d'enregistrement

        Arguments :
        * file : string, emplacement du fichier à processer 
        * play : boolean (optional), permet de jouer (sur les hauts parleurs) ou non le son 
        * frequence_enregistrement : float/int (optional), si l'on veut preciser une frequence de l'enregistrement 
        * stereo : boolean (optional), permet de preciser si le son est en stereo ou non

    * matrixComputing(son_array, frequence_enr, time_step, stereo)
        
        Fonction permettant d'obtenir les transformee successives de l'enregistrement ainsi que la duree totale de l'enregistrement

        Retour :
        * fft_mat : ndarray, matrice de la transformee de Fourier
        * T : float, duree totale de l'enregistrement

        Arguments :
        * son_array : ndarray
        * frequence_enr : float, precise la frequence de l'enregistrement
        * time_step : float, precise la duree sur laquelle on effectue les transformées de Fourier
        * stereo : boolean, precise la nature du son present dans la matrice, mono ou stereo

    * spectroPlotting(fft_mat, T, displayStretch, stereo, cmap)

        Fonction permettant d'afficher le spectrogramme d'une matrice déjà calculée dans un graphe puis de l'afficher

        Retour : None 
        
        Arguments :
        * fft_mat : ndarray, matrice a representer sur le graph
        * T : float, duree totale de l'enregistement 
        * displayStretch : float, permet de controler l'affichage vertical du graph (plus il est grand moins la frequence max affichee est grande)
        * cmap : string, permet de preciser la colormap a utiliser pour la representation du graph

    * reconstitution_son(fft_mat_output, frequence_enr, play=True, plot=True)

        Fonction permettant de reconstituer un son a partir d'une matrice
        
        Retour :
        * reconstitution : ndarray, contenant le son reconstitué

        Arguments : 
        * fft_mat_output : ndarray, matrice contentant la fft que l'on doit transformer en son
        * frequence_enr : float/int, fréquence de l'enregistrement à reconstituer
        * play : boolean (optionnal), permet de jouer ou non le son reconstitué
        * plot : boolean (optionnal), permet d'afficher ou non sur un graph le son reconstitué

