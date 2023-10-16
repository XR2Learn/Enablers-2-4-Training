# Features Extraction (Handcrafted) for Audio Modality

Component to pre-train an ED model using audio data input type.

`Input`: CSV file with labels + path to audio files + timestamp.

- CSV Input file should be in the folder `/datasets/<input_CSV_file.CSV>`

`Output`: CSV files (train.csv, test.csv, val.csv) with labels + path to features extracted, per features type. 

- Output files should be saved in the folder `/outputs/handcrafted-features-generation-audio/`

### Support
- This component does not support macOS with Apple chip because one of its dependency, opensmile,
does not support it. Check reported [issue](https://github.com/audeering/opensmile-python/issues/79#issuecomment-1614165695).

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

# TODO
 - change features to extract from listy to keys/dicts with parameters