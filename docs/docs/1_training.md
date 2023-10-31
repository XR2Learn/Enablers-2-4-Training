# XR2Learn-Training

Repository containing the components for training the models used in XR2Learn.

- Pre-processing: Pre-process raw signal data, for example, standardise, normalise.
- Handcrafted Features Extraction: Extracts MFCCs and eGeMAPS.
- Self-Supervised Learning (SSL) Training (pre-train): Pre-train an encoder.
- SSL Features Extraction: Uses a trained encoder to generate features and save to disk.
- Supervised Learning Training (fine-tuning): Train a classification model and save weights to disk.

[Diagram with Architecture Overview](https://drive.google.com/file/d/1k3yLi9Y8tasFMJFNxIwKY-nRJzPdKPLw/view?usp=sharing)

## Dependencies:

- Docker (Nvidia-Docker for CUDA)
- Python 3.10

## Changelog
[CHANGELOG.md]

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

### Running on GPU

1. Using Docker images

`docker compose -f docker-compose.yml -f docker-compose-gpu.yml run --rm <service-name>`

2. Local run

   Set up your local virtual environment using `requirements-gpu.txt` file instead of `requirements.txt`

## Folders configuration: /datasets and /output 

By default, to facilitating the development of multiple components, docker-compose.yml is configured to map the dockers
images folders
`\datasets` `\outputs` and the file `configuration.json` to a single location in the repository root's directory.

If you do not want a single `\datasets`, `\outputs` folder to all the docker images when running docker in development,
eliminate the volumes mapping by commenting the lines in `docker-compose.yml` file. For example:

`"./datasets:/app/datasets"`

`"./outputs:/app/outputs"`

`"./configuration.json:/app/configuration.json"`

Then, the docker images will map the `/datasets`, `/outputs` and `configuration.json` file from the ones inside each
component.

[CHANGELOG.md]: https://github.com/um-xr2learn-enablers/XR2Learn-Training/blob/master/CHANGELOG.md