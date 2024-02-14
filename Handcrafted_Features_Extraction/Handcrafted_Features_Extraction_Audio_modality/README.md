# Features Extraction (Handcrafted) for Audio Modality

Component to pre-train an ED model using audio data input type.

`Input`: CSV file with labels + path to audio files + timestamp.

- CSV Input file should be in the folder `/datasets/<input_CSV_file.CSV>`

`Output`: CSV files (train.csv, test.csv, val.csv) with labels + path to features extracted, per features type.

- Output files should be saved in the folder `/outputs/handcrafted-features-generation-audio/`

### Support

- The “Handcrafted Features Extraction” component for audio modality does not support macOS - Apple chip,
  due to the OpenSmile Python library not supporting this OS architecture, check
  reported [issue](https://github.com/audeering/opensmile-python/issues/79#issuecomment-1614165695).
  “Handcrafted Features Extraction” still supports macOS Intel chip machines.

# Development

## Development Environment Setup

1. Clone the repository
2. Prepare your virtual environment (e.g. VirtualEnv, Pip env, Conda)

## Development cycle (simplified version)

1. Crete a new branch to develop the task
2. Commit to remote to the new branch as needed
3. After finishing developing test docker image
4. When is everything done and tested, merge task branch to master branch

## Development Notes

## Configuration

1. For local run create `configuration.json` file in the same level as `example.configuration.json`

## Environment Variables (Env Vars)

1. For local run create `.env` file in the same level as `example.env` to load environment variables.
2. Add on docker compose the environment variable name under the service `ssl-audio`

## License

The handcrafted features extraction component for the audio modality is shared for non-commercial use only, to comply
with [OpenSMILE](https://github.com/audeering/opensmile-python) license.

Copyright © 2024, Maastricht University

Please refer to [LICENSE.md](LICENSE.md)
document for more details.
