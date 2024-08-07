# Pre-processing - Audio Modality

Component to pre-process raw data into an organized time window of data and labels to be used by the other
components.

`Input`: Raw files from a dataset

- Dataset Input files should be in the folder `/datasets/<input_CSV_file.CSV>`

`Output`: CSV file with labels + path to audio files + timestamp.

- CSV Output file should be in the folder `/datasets/<input_CSV_file.CSV>`

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


# Comments / To-Do
- check influence of outliers in min/max normalization
- create (jupyter) notebook to check preprocessing/handcrafted features on selected audio files

