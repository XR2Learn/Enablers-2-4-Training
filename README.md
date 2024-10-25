# [XR2Learn Personalization Enablers] Training Tools

Training tools involve Enablers 2, 3, and 4 with related components, a total of five tools, for pre-training and
fine-tuning models used in XR2Learn. Each tool is a modularized component with an isolated environment and dependencies
that can be used separately, in combination, or as an end-to-end system (together with
the [Command-Line Interface – CLI](https://github.com/XR2Learn/Enablers-CLI)).

The components are also separated by modalities, e.g., audio, bio measurements (BM), and body tracking modalities. Each component is
deployed using Docker to ensure easy-to-use components, reproducible development and deployment environments, and
consistent results. Thus, the Training tools support cross-platform use, i.e., Windows, Linux and macOS.

- **Pre-processing**: Pre-process raw data into an organized time window of data and labels to be used by the other
  components.
- **Handcrafted Features Extraction**: Extracts features derived from the raw data type’s properties instead of using
  Machine Learning for feature extraction.
- **Self-Supervised Learning (SSL) Training (pre-train)**: Pre-train an encoder (Enabler 2), with no use of labels.
- **SSL Features Extraction**: Uses an encoder to generate features (Enabler 3).
- **Supervised Learning Training (fine-tuning)**: Trains a classification model (Enabler 4) utilizing labels.

Pre-trained encoder and fine-tuned emotion classification models are also available for use.

[Diagram with Architecture Overview](https://drive.google.com/file/d/1k3yLi9Y8tasFMJFNxIwKY-nRJzPdKPLw/view?usp=sharing)

## Pre-requisites:

The Training Tools support the three main Operational Systems (OS): Linux, macOS, and Windows, as well as CPU and GPU
use.

The two pre-requisites are:

- Docker installed (or Docker-Nvidia if GPU use is required)
- Python 3.10 installed

## Installation

You do not need to install any additional application to run the Enablers and their components, they are deployed using
Docker containers, so you can access the Enablers and components by running docker commands.
You can find a list of useful docker compose commands below.

1. Navigate to the root directory of the downloaded project, and from the root repository, run the command to build the
   docker images

`docker compose build`

3. If using GPU, also build the docker images for GPU

`docker compose -f docker-compose.yml -f docker-compose-gpu.yml build` (for Windows systems)

`./compose-gpu.sh build` (for Unix based systems, i.e., Linux and MacOS)

For an easier interface to use Enablers' functionalities, please
check [Enablers-CLI](https://github.com/XR2Learn/Enablers-CLI) repository.

## Basic User Manual

The enablers were designed to be used with Enablers-CLI, a command-line interface that simplifies the use of enablers,
so the easiest way to access the Enablers’ functionalities is by using Enablers-CLI. Please refer to Section 3.5 for
information on how to use CLI. However, if changing or expanding the enablers’ functionalities is required, it is
possible to access each component using docker commands, as exemplified below. Thus, the instructions described below
are focused on running the enablers for a development environment.

A `configuration.json` file is required to provide the components with the necessary specifications for running. A default
version of “configuration.json” is provided and can be changed by the user.

Run a docker image:
docker compose run --rm <service-name>
Note: Service names can be found in the “docker-compose.yml” file in the project’s root folder. Each modality, i.e.,
audio, bm (bio-measurements), body-tracking, are deployed in separated docker containers and their service name follow
the structure:
pre-processing-<modality>
handcrafted-features-generation-<modality>
ssl-<modality>
ssl-features-generation-<modality>
ed-training-<modality>
There is an additional script to run all the docker images from a given modality, which will use the available
‘configuration.json’ file:
For Unix-based OS, macOS and Linux
./run_all_dockers-<modality>.sh
For Windows:
./run_all_dockers.ps1

### Additional Useful Commands:

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

`./run_all_dockers-<modality>.sh`

`CONFIG_FILE_PATH=<path-to-configuration-json> ./run_all_dockers-<modality>.sh`

All the outputs produced by any component in the Training domain are saved and can be accessed in the folder ./outputs.

### Running on GPU

1. Using Docker images

`docker compose -f docker-compose.yml -f docker-compose-gpu.yml run --rm <service-name>`

or

`./compose-gpu.sh run --rm <service-name>`

2. Local run

   Set up your local virtual environment using `requirements-gpu.txt` file instead of `requirements.txt`

## Changelog

For a detailed list of major changes in each version, please check in:
[CHANGELOG.md]

## License

Copyright © 2024, Maastricht University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The handcrafted features extraction component for the audio modality is shared for non-commercial use only, to comply
with [OpenSMILE](https://github.com/audeering/opensmile-python) license.

Pre-trained and fine-tuned models created using the RAVDESS dataset are shared under
the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license to
comply with the RAVDESS license, as the models are derivative works from this dataset.

Please refer to [LICENSE.md](LICENSE.md) document for more details.

### Contributors 
- University of Applied Sciences and Arts of Southern Switzerland (SUPSI): Body Tracking modality components

[CHANGELOG.md]: CHANGELOG.md