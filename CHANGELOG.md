# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Information to README files.

### Changed
- Code refactoring (PIP8).

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


[unreleased]: https://github.com/um-xr2learn-enablers/XR2Learn-Training/compare/v0.1.0...master
[0.1.0]: https://github.com/um-xr2learn-enablers/XR2Learn-Training/releases/tag/v0.1.0