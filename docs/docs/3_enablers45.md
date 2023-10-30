# Enablers 4/5

This page presents a technical guide on Enablers 4/5: installation, script execution and guidelines for contribution. 

The step-by-step description on how to set up the environment and run enablers scripts is also available in the README file of its github repository: [https://github.com/um-xr2learn-enablers/enablers45](https://github.com/um-xr2learn-enablers/enablers45)

## Installation
1. Install the Conda package manager: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Clone the repository.

3. Run the following commands:

```
conda env create -f environment.yml;

conda activate enablers45
```


## Enabler 4

Enabler 4 can be launched by a single command in terminal using various arguments that can be provided. A more detailed description of each argument can be found in the [README](https://github.com/um-xr2learn-enablers/enablers45/blob/main/README.md) of the enabler. 

Example of supervised learning (from scratch) 1D-CNN for bio-measurement data (EDA, BVP, Temp) from WESAD dataset which is located in `<WESAD_DATA_PATH>` (unzipped):

```
python train.py --dataset_config_path ./configs/datasets/bm/wesad.yaml --preprocessing_configs ./configs/pre-processing/bm/wesad_eda_bvp_temp.yaml --model_configs ./configs/models/bm/wesad_cnn1d_stacked.yaml --dataset wesad --data_path <WESAD_DATA_PATH>/WESAD/ --gpus 1 --num_workers 16 
```

Same model fine-tuning, given the frozen encoder of the same configuration pre-trained in Enablers 2/3:

```
python train.py --dataset_config_path ./configs/datasets/bm/wesad.yaml --preprocessing_configs ./configs/pre-processing/bm/wesad_eda_bvp_temp.yaml --model_configs ./configs/models/bm/wesad_cnn1d_stacked_frozen.yaml --dataset wesad --data_path /home/data/emotion_rec/WESAD/ --gpus 1 --num_workers 16 --pre_trained_paths <PRE_TRAINED_PATH>.ckpt
```

Training audio model using wav2vec2.0 feature extractor (WAV2VEC2 Base from https://pytorch.org/audio/0.10.0/pipelines.html) on IEMOCAP dataset:

```
python train.py --dataset_config_path ./configs/datasets/speech/iemocap.yaml --preprocessing_configs ./configs/pre-processing/speech/iemocap_wav2vec.yaml --model_configs ./configs/models/speech/iemocap_wav2vec.yaml --dataset iemocap --data_path <IEMOCAP_FOLDER>/IEMOCAP_full_release/ --gpus 1 --num_workers 16
```

???+ note
    `WESAD_DATA_PATH`:
        
        (UM) `cvi.dke.unimaas.nl` : `/home/data/emotion_rec/WESAD/`
        (UM) `avcl02.dke.unimaas.nl` : `/home/data/emotion_rec/WESAD`

    `IEMOCAP_DATA_PATH`:

        (UM) `cvi.dke.unimaas.nl` : `/home/data/emotion_rec/IEMOCAP_full_release/`

### Configuration Guide
Configurations used in Enablers 4/5 are similar and, partially, identical to the ones from Enablers 2/3. Similarly to Enablers 2/3, we present the configurations of the same dataset (WESAD) and encoder (CNN1D1L).

**Datasets**:

The following example is available at `configs/datasets/bm/wesad.yaml`. This configuration file can be exactly the same for both Enablers 2/3 and 4/5.

```
# pool of values that can be used for splitting data into train, validation and test. 
split_pool: ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13','S14', 'S15', 'S16'] 
# default protocol, used in supervised learning. For Enablers 2/3, this is an optional parameter.
default_protocol: "3-class" 
# description of the protocols. The listed protocols should be supported by associated classes in Emotion Recognition Toolbox.
protocols: 
  4-class:
    class_names: ["baseline", "stress", "amusement", "meditation"]
    n_classes: 4
  3-class:
    class_names: ["baseline", "stress", "amusement"]
    n_classes: 3
  2-class:
    class_names: ["no stress", "stress"]
    n_classes: 2
# main metric to report the performance. In the case of the SSL, only the loss is tracked, since the labels are not be available.
main_metric: "loss" 
```

**Pre-processing**:

The following example is available at `configs/pre-processing/bm/wesad_eda_bvp_temp.yaml`. This configuration file can be exactly the same for both Enablers 2/3 and 4/5.

```
# general pre-processing configs
general_pre_processing:
  # in seconds
  # dataset protocol to use, again required for Enablers 4/5
  dataset_protocol: '3-class'
  # length of data segments to be generated
  sequence_length: 10
  # type of fusion; supported: input_fusion and feature_fusion (needs testing)
  fusion_location: input_fusion
  # overlapping interval
  overlap : 5
  # flag for applying z-normalization
  normalize : True

# pre-processing for each sensor
sensors:
  # name of modality from the dataset
  wrist_eda:
    # list of transforms to be applied: should be implemented in emorec_toolbox/utils/general_transforms.py
    transforms:
      - class_name: Resample
        from_module: general_transforms
        transform_name: resample
        in_test: true
        kwargs:
          samples: 128
      - class_name: ToTensor
        from_module: general_transforms
        transform_name: to_tensor
        in_test: true
      - class_name: Permute
        from_module: general_transforms
        transform_name: permutation
        in_test: true
        kwargs:
          shape: [1, 0]
      - class_name: ToFloat
        transform_name: to_float
        from_module: general_transforms
        in_test: true
  wrist_bvp:
    transforms:
      - class_name: Resample
        from_module: general_transforms
        transform_name: resample
        in_test: true
        kwargs:
          samples: 128
      - class_name: ToTensor
        from_module: general_transforms
        transform_name: to_tensor
        in_test: true
      - class_name: Permute
        from_module: general_transforms
        transform_name: permutation
        in_test: true
        kwargs:
          shape: [1, 0]
      - class_name: ToFloat
        transform_name: to_float
        from_module: general_transforms
        in_test: true
  wrist_temp:
    transforms:
      - class_name: Resample
        from_module: general_transforms
        transform_name: resample
        in_test: true
        kwargs:
          samples: 128
      - class_name: ToTensor
        from_module: general_transforms
        transform_name: to_tensor
        in_test: true
      - class_name: Permute
        from_module: general_transforms
        transform_name: permutation
        in_test: true
        kwargs:
          shape: [1, 0]
      - class_name: ToFloat
        transform_name: to_float
        from_module: general_transforms
        in_test: true
```


**Models**:

The following example is available at `configs/models/bm/wesad_cnn1d1l_stacked.yaml`. 

```
# architecture of the supervised model from models/supervised_model.py. This is modality agnostic supervised framework.
architecture: StackedMultiModalClassifier 
# modalities to use
modalities: ['wrist_eda','wrist_bvp','wrist_temp']
# group modalities for encoders. Here, we stack all three modalities in one group
grouped: ['wrist_eda_bvp_temp']
# name of the model
models: ['cnn1d1l']
# type of training; use supervised in this enabler
framework: supervised
# name of the dataset
dataset: wesad

# experiment parameters
experiment:
  seed: 28
  num_epochs_fine_tuning: 150
  num_epochs_early_stopping: 100
  batch_size_fine_tuning: 64

# model architecture and hyperparameters: should be compatible with emorec_toolbox
# structure is exactly the same as in the Enabler 2
encoders:
  wrist_eda_bvp_temp:
      input: wrist_eda_bvp_temp
      enc_architecture:
        class_name: CNN1D1L
        from_module: models.bm.cnn1d
        encoder_name: wrist_eda_bvp_tmp
        args: []
        kwargs:
          in_channels: 3
          len_seq: 128
          out_channels: [16]
          kernel_sizes: [7]
          stride: 1
          padding: 0
          pool_padding: 0 
          pool_size: 2
          p_drop: 0.2

# classifier hyperparameters
classifier:
  # size of the fusion layer to map modalities
  fusion: 128
  # learning rate
  lr: 0.0001
  optimizer_name: "adam"
  metric_name: "f1-score"
  # splits (currently this parameter is not used, to be added)
  splits: random
  # if pre-trained model is used we can freeze the weights of the encoder
  freeze_encoders: False
  # type of the classification layer
  linear_eval: True  

```

## Enabler 5: 

TBC