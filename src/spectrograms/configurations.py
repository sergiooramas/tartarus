### Spectrograms
config_spectro = {
	'SUPER' : {
		'audio_folder' : '/homedtic/soramas/tartarus/dummy-data/audio/',
		'spectrograms_name' : 'SUPER',
		'resample_sr' : 22050,
		'hop' : 1024,
		'spectrogram_type' : 'cqt',
		'cqt_bins' : 96,
		'convert_id' : True, # converts the (path) name of a file to its ID name - correspondence in index_file.
		'index_file' : 'index_SUPER.tsv', # index to be converted. THIS IS THE LIST THAT ONE WILL COMPUTE
		'audio_ext' : ['mp3'] , # in list form
		'num_process' : 8,
		'compute_spectro' : True
	}
}

