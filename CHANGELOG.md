# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2024-02-15

### Added

- More Documentation
- License

## [0.3.1] - 2024-01-19

### Added

- Structure for the BM modality components (Preprocessing, Handcrafted Features, SSL Training, Supervised Training and
  SSL Features Extraction)

## [0.3.0] - 2023-12-06

### Added

- More documentation.
- Unit tests for Pre-processing, SSL and Handcrafted Features extraction pipelines for audio
- Support pipelines for custom audio datasets prepared in the provided format
- Support for W2V2 models in SSL features extraction
- Wav2Vec2 Large encoder

### Changed

- Handcrafted feature extraction logic: one pass through data to generate required features
- Refactoring for Pre-processing, SSL and Handcrafted Features extraction pipelines for audio
- Include features from local CNN encoder in Wav2Vec2

### Fixed

- ~~Only CNN model is supported for SSL feature extraction~~
- ~~W2V2 implementation: features from local CNN encoder are not included~~

### Known Issues

- `docker-compose.yml` is mapping outside .env file to docker image, generating an error in some cases.

## [0.2.0] - 2023-11-14

### Added

- Information to README files.
- Added tests for the augmentations implemented
- First version of documentation and API with mkdocs+mkdocstrings
- Validation and test performance logging with torchmetrics and `LogClassifierMetrics` callback
- Best (last if needed) model checkpointing using Pytorch Lightning
- Unit tests coverage for the Supervised Audio Modality Component

### Changed

- Code refactoring (PIP8).
- Convert all encoders and model to Pytorch Lightning (Supervised Component)
- Refactoring of augmentations.py into more descriptive files based on the type of data to augment
- CNN encoder now accepts variable length list inputs for channels/kernels hyperparameters

### Fixed

- ~~Passing `EXPERIMENT_ID` as an Env VAR is not supported for docker images.~~
- ~~Augmentations are not being applied and wrong augmentation logic in Datamodules~~
- ~~Model checkpointing: only last epoch saved~~
- ~~Limited logging (no performance metrics being logged)~~

### Known Issues

- `docker-compose.yml` is mapping outside .env file to docker image, generating an error in some cases.
- Only CNN model is supported for SSL feature extraction
- W2V2 implementation: features from local CNN encoder are not included

## [0.1.0] - 2023-10-19

### Added

- Basic version of Enablers 2-4 for audio modality using RAVDESS dataset, supporting:
    - Preprocessing with standardize
    - SSL Encoder with CNN and Wav2vec2
    - Handcrafted features with eGeMAPS and MFCCs
    - Supervised Training
    - SSL features extraction
- Changelog

### Known Issues

- `docker-compose.yml` is mapping outside .env file to docker image, generating an error in some cases.
- Passing `EXPERIMENT_ID` as an Env VAR is not supported for docker images.
- Augmentations are not being applied and wrong augmentation logic in Datamodules
- Model checkpointing: only last epoch saved
- Limited logging (no performance metrics being logged)
- Only CNN model is supported for SSL feature extraction

<!-- 
Example of Categories to use in each release

### Added
- Just an example of how to use changelog.

### Changed
- Just an example of how to use changelog.

### Fixed
- Just an example of how to use changelog.

### Removed
- Just an example of how to use changelog.

### Deprecated
- Just an example of how to use changelog. -->


[unreleased]: https://github.com/um-xr2learn-enablers/XR2Learn-Training/compare/v0.3.1...master

[0.1.0]: https://github.com/um-xr2learn-enablers/XR2Learn-Training/releases/tag/v0.1.0

[0.2.0]: https://github.com/um-xr2learn-enablers/XR2Learn-Training/releases/tag/v0.2.0

[0.3.0]: https://github.com/um-xr2learn-enablers/XR2Learn-Training/releases/tag/v0.3.0

[0.3.1]: /releases/tag/v0.3.1

[0.3.2]: /releases/tag/v0.3.1
