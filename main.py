import spectrogram

[fft_mat_output, frequence_enr ] = spectrogram.spectrogramme_wav('audio/LA_mono.wav', displayStretch=20, time_step=0.05, play=False, stereo=False)

spectrogram.reconstitution_son(fft_mat_output, frequence_enr,plot=True)
