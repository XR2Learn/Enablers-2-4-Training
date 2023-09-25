# Self-Supervised Learning Training - Audio Modality

Component to pre-train an ED model using audio data input type.

`Input`: CSV file with labels + path to audio files + timestamp.

- CSV Input file should be in the folder `/datasets/<input_CSV_file.CSV>`

`Output`: File with trained weights.

- Output file with trained weights is saved in the folder `/outputs/<file_weights>`

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

- Which configuration do we need to create this component?

## Configuration

1. For local run create `configuration.json` file in the same level as `example.configuration.json`

## Environment Variables (Env Vars)

1. For local run create `.env` file in the same level as `example.env` to load environment variables.
2. Add on docker compose the environment variable name under the service `ssl-audio`

# TODO
    -   restore/save best checkpoint
