# Pre-processing - Audio Modality

Component to pre-train an ED model using audio data input type.

`Input`: CSV file with labels + path to audio files + timestamp.

- CSV Input file should be in the folder `/datasets/<input_CSV_file.CSV>`

`Output`: File with trained weights.

- Output file with trained weights is saved in the folder `/outputs/<file_weights>`

# Development

## Development Environment Setup
Create and activate virtual environment with `venv`. If you wish to use GPU in your environment, install requirements from `requirements-gpu.txt` (line 3):
``` 
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements{-gpu}.txt
```

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

