### Spectrograms
config_spectro = {
	'MSD' : {
		'spectrograms_code_version': 'v0dev',
		'audio_folder' : '', 	# end it with / !!!
		'spectrograms_name' : 'MSD',
		'resample_sr' : 22050,
		'hop' : 1024,
		'spectrogram_type' : 'cqt',
		'cqt_bins' : 96,
		'convert_id' : False, 										# converts the (path) name of a file to its ID name - correspondence in index_file.
		'index_file' : 'index_MSD.tsv',				# index to be converted. THIS IS THE LIST THAT ONE WILL COMPUTE
		'audio_ext' : ['mp3'] , # in list form
		'num_process' : 8 ,
		'compute_spectro' : True
	}
}

