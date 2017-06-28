# Tartarus

Tartarus is a library for Deep Learning experiments on Audio and Text and their combination

To use this library you need to create a folder structure as follows

/tartarus
	/src
	/data
		/datasets
			/train_data
		/factors
		/models
		/results
		/patches

You need to put your dataset files inside data/datasets. Files that describe the dataset need to have the following nomenclature:

A file with index ids, with one id per line, called

items_index_$1_$2.tsv

$1 set (train, test)
$2 dataset name

Files with the factors obtained from matrix factorization

item_factors_$1_$2_$3.npy
user_factors_$1_$2_$3.npy

$1 kind of factorization (als, pmi)
$2 dimensions of the output
$3 dataset name

A file with the ground truth matrix for the test set

matrix_test_$1.npz

$1 dataset name


In addtion, in folder data/dataset/train_data we need to put the X matrices with the features to train the models. These files may be dense or sparse matrices. X files should have the following nomenclatures:

X_$1_$2_$3.npy/npz

$1 set (train, test)
$2 setting name
$3 dataset name

When using audio, in folder /data/patches we need to add h5py files with the spectrogram patches of the different sets. This files must have 2 datasets inside, one called index, where the item ids must be stored, and the other called features, where the spectrogram patches should be stored. Patches may be created using the scripts inside src/spectrograms folder. The nomenclature of patches files is the following:

patches_$1_$2_$3_$4.hdf5

$1 set name
$2 dataset name
$3 number of patches per item
$4 seconds per patch


To run experiments you can simply excecute the run_experiments.py script, passing the experiment name and the setting name by arguments.

You can also run train, prediction and evaluation separately.

It is possible to obtain a specific layer prediction output from predict.py by using the -l parameter.

## DLRS Experiments

To reproduce the experiments in the Cold-start recommendation paper, you have to download the dataset from and untar it in the /tartarus folder.

Then you can run the experiments by calling run_experiments.py, for example:

For artist recommendation:

python run_experiments.py rec_sparse sem-bio

For multimodal recommendation:

python run_experiments.py rec_multi a-sem-emb audio-emb

More approaches from the paper can be tested modifying the configuration inside run_experiments.py

