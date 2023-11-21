# Enablers 2-4 Training

Repository containing Enablers 2-4 and components for pre-train and fine-tuning models used in XR2Learn.

- Pre-processing: Pre-process raw data, for example, standardise, normalise.
- Handcrafted Features Extraction: Extracts MFCCs and eGeMAPS.
- Self-Supervised Learning (SSL) Training (pre-train): Pre-train an encoder (Enabler 2).
- SSL Features Extraction: Uses a trained encoder to generate features (Enabler 3).
- Supervised Learning Training (fine-tuning): Train a classification model (Enabler 4).

[Diagram with Architecture Overview](https://drive.google.com/file/d/1k3yLi9Y8tasFMJFNxIwKY-nRJzPdKPLw/view?usp=sharing)

## Dependencies:

- Docker (Nvidia-Docker for CUDA)
- Python 3.10

## Installation



## Changelog
[CHANGELOG.md]

## Documentation 


## Docker Commands

1. Build a docker image

`docker compose build <service-name>`

2. Run a docker image

`docker compose run --rm <service-name>`

3. Run a docker image giving EnvVars

`KEY=VALUE docker compose run --rm <service-name>`

4. Run a docker image with shell entrypoint

`docker compose run --rm <service-name> \bin\bash`

5. Run all dockers from the Training Domain (The script deletes contents from `/datasets` and `/outputs` folders and run
   all
   docker images)

`./run_all_dockers.sh`

`CONFIG_FILE_PATH=<path-to-configuration-json> ./run_all_dockers.sh` 

### Running on GPU

1. Using Docker images

`docker compose -f docker-compose.yml -f docker-compose-gpu.yml run --rm <service-name>`

or

`./compose-gpu.sh run --rm <service-name>`

2. Local run

   Set up your local virtual environment using `requirements-gpu.txt` file instead of `requirements.txt`



[CHANGELOG.md]: https://github.com/um-xr2learn-enablers/XR2Learn-Training/blob/master/CHANGELOG.md