import argparse
import os
import librosa
from joblib import Parallel, delayed
import pickle
import json
import sys
sys.path.insert(0, '../')
import common
from configurations import config_spectro
import signal

"""
spectrograms.py: computes spectrograms.

Requires pre-precomputing an 'index_file',a .tsv where an index with id,path is defined for a dataset.

The results and parameters of this script are stored in common.DATA_PATH/spectrograms/
'index.tsv' stores the 'id,path_spectrogram,path_audio'.
'path_spectrgram' and 'path_audio' absolute reference from common.DATA_PATH.

Step 1/5 of the pipeline.

"""

# Choi et al.: 	'original_sr' : 16000, 'resample_sr' : 12000, 'hop' : 256, 'spectrogram_type' : 'mel', 'n_fft' : 512, 'n_mels' : 96.
# Dieleman et al.: 	'original_sr' : 16000, 'resample_sr' : 16000, 'hop' : 256, 'spectrogram_type' : 'mel', 'n_fft' : 512, 'n_mels' : 128.

path2id = dict()
config = dict()

def signal_handler(signum, frame):
    raise Exception("Timed out!")

def compute_spec(audio_file,spectro_file):
	# Get actual audio
	audio, sr = librosa.load(audio_file, sr=config['resample_sr'])
	# Compute spectrogram
	if config['spectrogram_type']=='cqt':
		spec = librosa.cqt(audio, sr=sr, hop_length=config['hop'], n_bins=config['cqt_bins'], real=False)
	elif config['spectrogram_type']=='mel':
		spec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=config['hop'],n_fft=config['n_fft'],n_mels=config['n_mels'])
	elif config['spectrogram_type']=='stft':
		spec = librosa.stft(y=audio,n_fft=config['n_fft'])
	# Write results:
	with open(spectro_file, "w") as f:
		pickle.dump(spec, f, protocol=-1) # spec shape: MxN.

def do_process(id, audio_file, spectro_file):
	try:
		if config['compute_spectro']:
			if not os.path.exists(spectro_file[:spectro_file.rfind('/')+1]):
				os.makedirs(spectro_file[:spectro_file.rfind('/')+1])
			if not os.path.isfile(spectro_file): 
				signal.signal(signal.SIGALRM, signal_handler)
				signal.alarm(10)
				compute_spec(audio_file,spectro_file)
			fw = open(common.SPECTRO_PATH+config['spectro_folder']+"index.tsv","a")
			fw.write("%s\t%s\t%s\n" % (id,spectro_file[len(common.SPECTRO_PATH):],audio_file[len(common.AUDIO_PATH):]))
			fw.close()
			print 'Computed spec: %s' % audio_file
		else:
			if os.path.isfile(spectro_file):
				fw = open(common.SPECTRO_PATH+config['spectro_folder']+"index.tsv","a")
				fw.write("%s\t%s\t%s\n" % (id,spectro_file[len(common.SPECTRO_PATH):],audio_file[len(common.AUDIO_PATH):]))
				fw.close()
	except Exception as e:
		ferrors = open(common.SPECTRO_PATH+config['spectro_folder']+"errors.txt","a")
		ferrors.write(audio_file+"\n")
		ferrors.write(str(e))
		ferrors.close()
		print 'Error computing spec', audio_file
		print str(e)

def process_files(files):
	Parallel(n_jobs=config['num_process'])(delayed(do_process)(id, audio_file, spectro_file)
							   for id, audio_file, spectro_file in files)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Create spectrograms',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('configuration', default="magna", help='Id of the configuration dictionary')
	args = parser.parse_args()
	config = config_spectro[args.configuration]

	# set spectrograms folder
	if config['compute_spectro']:
		config['spectro_folder'] = "spectro_%s_%s/" % (config['spectrograms_name'],config['spectrogram_type'])
		print config['spectro_folder']
		if not os.path.exists(common.SPECTRO_PATH+config['spectro_folder']):
			os.makedirs(common.SPECTRO_PATH+config['spectro_folder'])
		#else:
		#	sys.exit("EXIT: already exists a folder with this name!\nIf you need to compute those again, remove folder.")

	# create empty spectrograms index
	fw = open(common.SPECTRO_PATH+config['spectro_folder']+"index.tsv","w")
	fw.close()

	# list audios to process: according to 'index_file'
	files_to_convert = []
	f=open(common.INDEX_PATH+config["index_file"])
	for line in f.readlines():
		id, audio = line.strip().split("\t")
		if config['convert_id']:
			spect = id+".pk"
		else:
			spect = audio[:audio.rfind(".")]+".pk"
		files_to_convert.append((id,common.AUDIO_PATH+config['audio_folder']+audio,common.SPECTRO_PATH+config['spectro_folder']+spect))

	print str(len(files_to_convert))+' audio files to process!'

	# compute spectrogram
	process_files(files_to_convert)

	# save parameters
	json.dump(config, open(common.SPECTRO_PATH+config['spectro_folder']+"params.json","w"))

	print "Spectrograms folder: "+common.SPECTRO_PATH+config['spectro_folder']

# COMMENTS:

## pickle protocol=-1?

## convert_id == FALSE: creates sub-directories - put to false for magna.
## convert_id == TRUE: does not creates sub-directories - in some cases one does not care.