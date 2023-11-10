## Documentation
```
{
  "dataset_config": {
    "dataset_name": name of dataset, should match folder name,
    "number_of_labels": number of labels in the dataset
  },
  "pre_processing_config": {
    "process": preprocessing to apply, "standardize" or "normalize"
    "create_splits": whether to create train.csv/test.csv/val.csv or not
    "target_sr": samplerate to resample to,
    "padding": add zero-padding or not, only works when max_lenght is defined
    "max_length": desired maximum lenght
  },
  "handcrafted_features_config": {
    "feature_name": kwargs
  },
  "encoder_config": {
    "from_module": module where the encoder is to be found,
    "class_name": name of encoder inside the previously specified module,
    "input_type": specify the input modality,should math the map name in the outputs folder,
    "kwargs": arguments for the encoder
    "pretrained_same_experiment": true
  },
  "ssl_config": {
    "from_module": module where the ssl method is to be found,
    "ssl_framework": name of the framework to use,
    "epochs": number of SSL training epochs,
    "batch_size": batch size for SSL training,
    "kwargs": other arguments for SSL training
  },
  "sup_config": {
    "epochs": number of supervised training epochs,
    "batch_size": batch size for supervised training,
    "use_augmentations_in_sup": whether to use the defined augmentations for supervised training,
    "kwargs": other supervised learning args
  },
  "augmentations": {
    "augmentation_name": {
      "probability": probability of augmentaion to be applied,
      "kwargs":
    },
    "augmentation_name": {
      "probability": probability of augmentaion to be applied,
      "kwargs":
    },
  },
  "transforms": [
    {
      "class_name": name of transform to apply,
      "from_module": where to find the transformation,
      "transform_name": name of transformation,
      "in_test": if transofrmation is to be applied to test set or not
    },
  ]
}
```

## Example
```
{
  "dataset_config": {
    "dataset_name": "RAVDESS",
    "number_of_labels": 8
  },
  "pre_processing_config": {
    "process": "standardize",
    "create_splits": true,
    "target_sr": 16000,
    "padding": true,
    "max_length": 5
  },
  "handcrafted_features_config": {
    "MFCC": {
      "sample_rate": 16000,
      "n_mfcc": 13,
      "melkwargs": {
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 23,
        "center": false
      }
    },
    "eGeMAPs": {
      "sampling_rate": 16000
    }
  },
  "encoder_config": {
    "from_module": "encoders.cnn1d",
    "class_name": "CNN1D",
    "kwargs": {
      "in_channels": 1,
      "len_seq": 88,
      "out_channels": [2,2,2],
      "kernel_sizes": [3,3,3],
      "stride": 2
    },
    "pretrained_same_experiment": true
  },
  "ssl_config": {
    "from_module": "ssl_methods.VICReg",
    "ssl_framework": "VICReg",
    "input_type": "eGeMAPs",
    "epochs": 20,
    "batch_size": 128,
    "kwargs": {
      "lr": 0.0001,
      "n_views": 2,
      "temperature": 0.05,
      "optimizer_name_ssl": "adam"
    }
  },
  "sup_config": {
    "epochs": 20,
    "batch_size": 128,
    "input_type": "eGeMAPs",
    "use_augmentations_in_sup": true,
    "kwargs": {
      "lr": 0.0001,
      "optimizer_name": "adam",
      "freeze_encoder": false
    }
  },
  "augmentations": {
    "gaussian_noise": {
      "probability": 0.33,
      "kwargs": {
        "mean": 0,
        "std": 0.2
      }
    },
    "scale": {
      "probability": 0.4,
      "kwargs": {
        "max_scale": 1.3
      }
    }
  },
  "transforms": [
    {
      "class_name": "ToTensor",
      "from_module": "general_transforms",
      "transform_name": "to_tensor",
      "in_test": true
    },
    {
      "class_name": "Permute",
      "from_module": "general_transforms",
      "transform_name": "permutation",
      "in_test": true,
      "kwargs": {
        "shape": [
          1,
          0
        ]
      }
    },
    {
      "class_name": "ToFloat",
      "from_module": "general_transforms",
      "transform_name": "to_float",
      "in_test": true
    }
  ]
}
```
