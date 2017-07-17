# Tartarus

Tartarus is a python module for Deep Learning experiments on Audio and Text and their combination.

Requirements: This library works with keras 1.1.0.

To use this library you need to create a folder structure as follows:

    /tartarus
        /src
        /data
            /datasets
                /train_data
            /factors
            /models
            /results
            /patches

You need to put your dataset files inside `data/datasets`. Files that describe the dataset need to have the following nomenclature:

A file with index ids, with one id per line, called:

    items_index_$1_$2.tsv

- $1 set (train, test)
- $2 dataset name

Files with the factors obtained from matrix factorization:

    item_factors_$1_$2_$3.npy
    user_factors_$1_$2_$3.npy

- $1 kind of factorization (als, pmi)
- $2 dimensions of the output
- $3 dataset name

A file with the ground truth matrix for the test set

    matrix_test_$1.npz

- $1 dataset name

In addition, in the folder `data/dataset/train_data` we need to put the X matrices with the features to train the models. 
These files may be dense or sparse matrices. X files should have the following nomenclatures:

    X_$1_$2_$3.npy/npz

- $1 set (train, test)
- $2 setting name
- $3 dataset name

When using audio, in folder `/data/patches` we need to add h5py files with the spectrogram patches of the different sets. 
This files must have 2 datasets inside, one called index, where the item ids must be stored, and the other called features, where the spectrogram patches should be stored. 
Patches may be created using the scripts inside `src/spectrograms` folder. 
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

Audio:
	python create_spectrograms.py SUPER
	python create_patches.py

Text:
	python load_vsm.py
	python load_w2v.py

### Experiment

Audio:
	python run_experiments.py dummy_audio

Text:
	python run_experiments.py dummy_text_vsm bow
	python run_experiments.py dummy_text_w2v w2v

### Prediction of feature embeddings

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

    https://drive.google.com/file/d/0B-oq_x72w8NUSGFZRzJmQXhYSlE/view?usp=sharing

Untar it and point DATA_DIR in common.py to the full path of the dlrs-data/ folder

Then you can run the experiments by calling `run_experiments.py`, for example:

For artist recommendation:

    python run_experiments.py rec_sparse sem-bio

For multimodal recommendation:

    python run_experiments.py rec_multi a-sem-emb audio-emb

More approaches from the paper can be tested modifying the configuration inside `run_experiments.py`.

Full dataset and description: 
	http://mtg.upf.edu/download/datasets/msd-a

## ISMIR 2017 Experiments (Multi-label Classification)

        Oramas S., Nieto O., Barbieri F., & Serra X. (2017) Multi-label Music Genre Classification from Audio, Text, and Images Using Deep Features. In Proceedings of the 18th International Society of Music Information Retrieval Conference (ISMIR 2017).

To reproduce the experiments in the Multi-label classification paper, you have to download the MuMu dataset and untar it.

This dataset contains the item-class matrices, data splits, and learned feature embeddings.

    https://

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

Full dataset and description: 
	https://www.upf.edu/en/web/mtg/mumu
