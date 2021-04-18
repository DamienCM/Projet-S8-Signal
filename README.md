# Projet

Mini Projet de Signal S8 sur la dissimulation d'images dans un son au travers de son spectrogramme

---------------------
## Initialisation du projet et installation des dépendances

Création d'un environnement virtuel :

1. Lancer un terminal (powershell sur windows)
2. Se placer dans le repertoire de travail avec la commande cd (ie le repertoire Projet)
3. Lancer les commandes suivantes afin de creer un environement virtuel, d'installer les dependances et de l'activer
    * Sur windows
   > $ python -m venv venv

   > $ . venv/Scripts/activate.ps1

   > $ pip install -r requirements.txt
    * Sur Ubuntu et systèmes Linux basés sur Ubuntu
   > $ sudo apt-get install python3-venv

   > $ sudo apt-get install python3-tk

   > $ python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

## Constitution du projet

Le projet est constitué de trois sous dossiers et de plusieurs scripts et librairies :

* audio :  
  Ce dossier contient plusieurs fichiers audio permettant de tester nos scripts.

* images :
  Ce dossier contient les images de travail à cacher dans le fichier audio et les images intermédiaires générées par le script.

* figures :
  Ce dossier contient les figures générées pour le rapport.

* spectrogram.py

  Il s'agit de la librairie principale du projet dans laquelle sont repertoriées la majorité des fonctions que nous utiliserons.

  Les fonctions principales (liste non exhaustive) sont:
  
  * > get_sound(file, play=False, frequence_enregistrement=None)
   
    Permet de récupérer le son depuis un fichier audio
  
  * > matrix_computing_sautant(son_array, frequence_enr, time_step, plot=False, ponderation=False, color_sep=False, ponderation_type='gaussian')
  
    Permet de calculer les matrices successives sautantes de notre signal
  
  * > matrix_computing_glissant(son_array, frequence_enr, time_step, ponderation=False)
  
    Permet de calculer les matrices successives glissantes de notre signal
  
  * > spectro_plotting(fft_mat, freq=1, displayStretch=2, title="Spectrogramme", cmap='Blues')
    
    Permet de tracer le spectrogramme à partir de la matrice des FFT
  
  * > reconstitution_son(fft_mat_output, frequence_enr=44000, play=False, plot=False)
    
    Permet de reconstituter le son à partir de la matrice des FFT
  
  * > addition_image_fft(image_path, fft_son, amplitude=1., x_scale=.5, x_shift=0., y_scale=.5, y_shift=0.)
  
    Permet d'ajouter l'image à la matrice des FFT du son
  
  * > addition_image_fft_colored(image, ffts)
    
    Permet d'ajouter une image colorée (3 matrices grayscale décalées dans le temps) à la matrice des FFT du son
  
  * > save_file(son_array, frequence, path='audio/out.wav')
  
    Permet de sauvegarder dans un fichier audio le son reconstitué
  
  * > re_assemblage_rgb(matrices_fft)
    Permet de réassembler les images RGB

