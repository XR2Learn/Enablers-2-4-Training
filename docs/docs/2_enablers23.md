# Enablers 2/3
This page presents a technical guide on Enablers 2/3: installation, script execution and guidelines for contribution. 

The step-by-step description on how to set up the environment and run enablers scripts is also available in the README file of its github repository: [https://github.com/um-xr2learn-enablers/enablers23](https://github.com/um-xr2learn-enablers/enablers23)

## Installation
1. Install the Conda package manager: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Clone the repository.

3. Run the following commands:

```
conda env create -f environment.yml;

conda activate enablers23
```


## Enabler 2
Enabler 2 can be launched by a single command in terminal using various arguments that can be provided. A more detailed description of each argument can be found in the [README](https://github.com/um-xr2learn-enablers/enablers23/blob/main/README.md) of the enabler. 

Example of SSL pre-training with the SimCLR framework using bio-measurements (EDA, BVP, Temp) from WESAD dataset which is located in <WESAD_DATA_PATH> (unzipped):

```
python pre-train.py --dataset_config_path ./configs/datasets/bm/wesad.yaml --preprocessing_configs ./configs/pre-processing/bm/wesad_eda_bvp_temp.yaml --model_configs ./configs/models/bm/wesad_cnn1d1l_stacked_simclr.yaml --dataset wesad --data_path <WESAD_DATA_PATH>/WESAD/ --gpus 1 --outputs_path ./enabler3_pretrained_models/
```

???+ note
    `WESAD_DATA_PATH`:
        
        (UM) `cvi.dke.unimaas.nl` : `/home/data/emotion_rec/WESAD/`

In this guide, we provide a more detailed description on how to define custom configurations for datasets, pre-processing, and models.

### Configuration Guide
Here, we present a set of configuration files in YAML based on the WESAD dataset:

**Datasets**:

The following example is available at `configs/datasets/bm/wesad.yaml`.

```
# pool of values that can be used for splitting data into train, validation and test. Needs a workaround if not labeled data is available in SSL.
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

The following example is available at `configs/pre-processing/bm/wesad_eda_bvp_temp.yaml`.

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

The following example is available at `configs/models/bm/wesad_cnn1d1l_stacked_simclr.yaml`.

```
# name of the SSL framework and its location in the current repository
ssl_framework: SimCLR
from_module: models.bm.simclr
# modalities to use
modalities: ['wrist_eda','wrist_bvp','wrist_temp']
# group modalities for encoders. Here, we stack all three modalities in one group
grouped: ['wrist_eda_bvp_temp']
# name of the model
models: ['cnn1d1l']
# type of training; use ssl in this enabler
framework: ssl
# name of the dataset
dataset: wesad

# experiment parameters
experiment:
  num_epochs_pre_training: 100
  batch_size_pre_training: 128

# model architecture and hyperparameters: should be compatible with emorec_toolbox
encoders:
  # should match grouped
  wrist_eda_bvp_temp:
      input: wrist_eda_bvp_temp
      # encoder architecture
      enc_architecture:
        # encoder is in ERT: emorec_toolbox/models/bm/cnn1d.py - CNN1D1L class
        class_name: CNN1D1L
        from_module: models.bm.cnn1d
        encoder_name: wrist_eda_bvp_tmp
        args: []
        # hyperparameters of the encoder.
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

# parameters of ssl;
ssl_setup:
  optimizer_name: "adam"
  kwargs:
    lr: 0.0001
    n_views: 2
    temperature: 0.05

# augmentation to be applied
# should be implemented in emorec_toolbox/utils/augmentations.py
augmentations:
  gaussian_noise:
    parameters:
      mean: 0
      std: 0.2
  scale:
    parameters:
      max_scale: 1.3
  time_shifting:
    parameters:
      max_shift: 32

```

## Enabler 3

TBC

