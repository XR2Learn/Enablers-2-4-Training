![XR2Learn](https://raw.githubusercontent.com/XR2Learn/.github/5c0fada6136915b389c1cd2151a0dd2cfc4a5aac/images/XR2Learn%20logo.png)

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

You do not need to install any additional application to run the Enablers and their components, they are deployed using
Docker containers, so you can access the Enablers and components by running docker commands.
You can find a list of useful docker compose commands below.

For an easier interface to use Enablers' functionalities, please
check [Enablers-CLI](https://github.com/XR2Learn/Enablers-CLI) repository.

## Changelog

For a detailed list of major changes in each version, please check in:
[CHANGELOG.md]

## Documentation

The documentation relative to this project can be found on the Wiki page of this repository.

More in-depth documentation can be found in this repository's `docs` folder.

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

## License

The Training tools code is shared under a dual-licensing model. For non-commercial use, it is released under the MIT
open-source license. A commercial license is required for commercial use.

The handcrafted features extraction components for the audio modality is shared for non-commercial use only, to comply
with [OpenSMILE](https://github.com/audeering/opensmile-python) license.

Pre-trained and fine-tuned models created using the RAVDESS dataset are shared under
the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license to
comply with the RAVDESS license, as the models are derivative works from this dataset.

Please refer to [LICENSE.md](LICENSE.md) document for more details.


[CHANGELOG.md]: CHANGELOG.md