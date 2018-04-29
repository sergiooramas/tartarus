# Tartarus

Tartarus is a python module for Deep Learning experiments on Audio and Text and their combination. It works for multiclass and multi-label classification, and for recommendation using matrix factorization techniques.

In this documentation 3 experiments are described.

* Test experiments to understand the executation pipeline (SUPER dataset)
* Recommendation experiments from the [DLRS-RecSys 2017 paper](http://mtg.upf.edu/node/3804)
* Multi-label classification experiments from the [ISMIR 2017 paper](http://mtg.upf.edu/node/3803)

Requirements: 
This library works with Keras deep learning framework and Theano backend.
To work with audio you will need also the librosa audio library.
There is a requirements.txt file with all library requirements to run it.

Once Keras is installed, you have to set up Theano backend and ordering in the .keras/keras.json config file in your home directory:

"image_data_format": "channels_first"
"backend": "theano"

If you want to use Tensorflow you have to change ordering in convolutions in src/models.py.

To use this library you need to create a folder structure as follows:

    /tartarus
        /src
        /data
            /splits
            /train_data
            /factors
            /models
            /results
            /patches

You need to put your dataset files inside `data/splits`. Files that describe the dataset need to have the following nomenclature:

A file with index ids, with one id per line, called:

    items_index_$1_$2.tsv

- $1 set (train, test)
- $2 dataset name

Files with the factors obtained from matrix factorization:

    y_$1_$2_$3.npy
    user_factors_$1_$2_$3.npy

- $1 kind of factorization (als, pmi)
- $2 dimensions of the output
- $3 dataset name

If the ground truth is for classification it is necessary two files:

    y_class_$2_$3.npy
    genre_labels_$3.npy

A file with the ground truth matrix for the test set

    matrix_test_$1.npz

- $1 dataset name

In addition, in the folder `data/train_data` we need to put the X matrices with the features to train the models. 
These files may be dense or sparse matrices. X files should have the following nomenclatures:

    X_$1_$2_$3.npy/npz

- $1 set (train, test)
- $2 setting name
- $3 dataset name

When using audio, in folder `/data/patches` we need to add h5py files with the spectrogram patches of the different sets. 
This files must have 2 datasets inside, one called index, where the item ids must be stored, and the other called features, where the spectrogram patches should be stored. 
Patches may be created using the scripts inside `src/audio-processing` folder. 
The nomenclature of patches files is the following:

    patches_$1_$2_$3_$4.hdf5

- $1 set name
- $2 dataset name
- $3 number of patches per item
- $4 seconds per patch

To run experiments you can simply execute the `run_experiments.py` script, passing the experiment name and the setting name by arguments.

You can also run train, prediction and evaluation separately.

It is possible to obtain a specific layer prediction output from `predict.py` by using the `-l` parameter.

## TEST Experiments

To test the library you can use this toy examples made with the SUPER dataset. This dummy dataset is made with 9 audio tracks, song lyrics, and 3 genre clases. To avoid copyright issues this dataset contains music of my own band Supertropica. Evaluations on this mini dataset has no sense, this is just an example to understand the full pipeline of the library.

### SUPER dataset

It is in folder dummy-data/
Open commons.py and set the full path of the dummy-data/ folder

	DATA_DIR=path to dummy-data folder

### Preprocessing data

Audio (run from audio-processing/ folder):

	python create_spectrograms.py SUPER
	python create_patches.py

Text (run from text-processing/ folder):

	python load_vsm.py
	python load_w2v.py

### Experiment

Audio:

	python run_experiments.py dummy_audio

Text:

	python run_experiments.py dummy_text_vsm bow
	python run_experiments.py dummy_text_w2v w2v

### Prediction of feature embeddings

Model names are assigned iteratively. I assume you have run the SUPER scripts with a clean installation of tartarus and in the order they are in this readme. Doing so, model_1 should be the audio model and model_2 the text bow model. When you train a model it will show you the model name before the training start.

Audio: 

	python predict.py model_1 -p -l 9 -s train
	python predict.py model_1 -p -l 9 -s val
	python predict.py model_1 -p -l 9 -s test

Text: 

	python predict.py model_2 -l 5 -s train
	python predict.py model_2 -l 5 -s val
	python predict.py model_2 -l 5 -s test


### Multimodal experiment

	python run_experiments.py dummy_multimodal model_1-pred_9 model_2-pred_5


## DLRS-RecSys 2017 Experiments (Recommendation)

Oramas S., Nieto O., Sordo M., & Serra X. (2017) A Deep Multimodal Approach for Cold-start Music Recommendation. https://arxiv.org/abs/1706.09739

To reproduce the experiments in the Cold-start recommendation paper, you have to download the MSD-A dataset and untar it.

This dataset contains the user-item matrices, factor matrices from the factorization, data splits, and learned feature embeddings.

[https://doi.org/10.5281/zenodo.831348](https://doi.org/10.5281/zenodo.831348)

Untar it and point DATA_DIR in common.py to the full path of the dlrs-data/ folder

Then you can run the experiments by calling `run_experiments.py`, for example:

For artist recommendation:

    python run_experiments.py rec_sparse sem-bio

For multimodal recommendation:

    python run_experiments.py rec_multi a-sem-emb audio-emb

More approaches from the paper can be tested modifying the configuration inside `run_experiments.py`.

Full dataset and description: 
	http://mtg.upf.edu/download/datasets/msd-a

## ISMIR 2017 and TISMIR Experiments (Multi-label Classification)

Oramas S., Nieto O., Barbieri F., & Serra X. (2017) Multi-label Music Genre Classification from Audio, Text, and Images Using Deep Features. In Proceedings of the 18th International Society of Music Information Retrieval Conference (ISMIR 2017).

To reproduce the experiments in the Multi-label classification paper, you have to download the MuMu dataset and untar it.

This dataset contains the item-class matrices, data splits, and learned feature embeddings.

[https://doi.org/10.5281/zenodo.831189](https://doi.org/10.5281/zenodo.831189)

Untar it and point DATA_DIR in common.py to the full path of the ismir-data/ folder

Then you can run the experiments by calling `run_experiments.py`, for example:


Multimodal experiments with LOGISTIC loss:

	python run_experiments.py logistic_multilabel_bi model_audio model_text
	python run_experiments.py logistic_multilabel_bi model_visual model_text
	python run_experiments.py logistic_multilabel_bi model_visual model_audio

Multimodal experiments with COSINE loss:

	python run_experiments.py cosine_multilabel_bi model_audio model_text
	python run_experiments.py cosine_multilabel_bi model_visual model_text
	python run_experiments.py cosine_multilabel_bi model_visual model_audio

Multimodal experiments with 3 modalities:

	python run_experiments.py logistic_multilabel_tri model_audio model_text model_visual

	python run_experiments.py cosine_multilabel_tri model_audio model_text model_visual

Text experiments:

	python run_experiments.py cosine_text_multilabel VSM
	python run_experiments.py cosine_text_multilabel VSM-SEM
	python run_experiments.py logistic_text_multilabel VSM
	python run_experiments.py logistic_text_multilabel VSM-SEM


Full dataset and description: 
	https://www.upf.edu/en/web/mtg/mumu


## TISMIR Experiments (Single-label Classification)

Oramas S., Barbieri F., Nieto O. & Serra X. (2017) Multimodal Deep Learning for Music Genre Classification. TISMIR.

To reproduce the experiments in the multimodal classification journal paper, you have to download the MSD-I dataset and untar it.

This dataset contains the item-class matrices, data splits, and learned feature embeddings.

[https://doi.org/10.5281/zenodo.831189](https://doi.org/10.5281/zenodo.831189)

Untar it and point DATA_DIR in common.py to the full path of the msdi-data/ folder

Then you can run the experiments by calling `run_experiments.py`, for example:


Experiments:

	python run_experiments.py single-label-vector audio
	python run_experiments.py single-label-multimodal audio visual
	python run_experiments.py single-label-multimodal audio mm-audio
	python run_experiments.py single-label-all audio visual mm-audio mm-visual


Full dataset and description: 
	https://www.upf.edu/en/web/mtg/msdi