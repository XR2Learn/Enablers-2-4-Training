# Emotion Recognition Toolbox
This page provides a guide on how to contribute to the Emotion Recognition Toolbox (ERT): setup the environemnt, work on existing and new components and test them. The description of the ERT components is also available in the README file of its github repository: [https://github.com/um-xr2learn-enablers/emorec-toolbox](https://github.com/um-xr2learn-enablers/emorec-toolbox)

## Setup Environment
Emotion Recognition Toolbox uses [poetry](https://python-poetry.org/docs/) to manage dependencies and build the project. In order to setup the environment to contribute to the toolbox, run the following commands:
```
git clone https://github.com/um-xr2learn-enablers/emorec-toolbox.git;

pip install poetry;

poetry install
```

After this step, the virtual environment with all the dependencies is built. In order to activate the environment, run:
```
poetry shell
```

You can also run the commands (e.g., tests) outside of the environment with:
```
poetry run <YOUR_COMMAND_HERE>
```

## ERT Components

The components of the toolbox are located in `emorec_toolbox/` folder. Currently, they include:

- `datasets/` : pre-processing tools for open-source (WESAD, IEMOCAP) and custom datasets (to be added). The pre-processing routines for each dataset contain raw data pre-processing and segmentation based on provided configurations (segment lenght, overlap, sampling rate) and associated labels (if available). Besides, for each dataset, the [PyTorch Dataset and Dataloder](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and [PyTorch-Lightning DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) classes have to be implemented for a smooth integration into the Enablers.

- `models/` : PyTorch models for each data modality.

- `utils/` : Augmentations, transformations and additional pre-processing utils for different modalities.

## Tests

The tests are designed with the [`unittest`](https://docs.python.org/3/library/unittest.html) library. In order to execute all tests with activated environment:
```
python -m unittest
```

You can also run tests outside the poetry environment:
```
poetry run python -m unittest
```


???+ note 
    In order to execute tests with data from `tests/test_datamodules.py`, make sure you adjust global variables with your local paths to datasets (line 12). Besides, for some datasets these tests can take more than 10 minutes.

You can also run specific tests as follows:
```
poetry run python -m unittest.<MODULE>.<TEST_CASE>
```

Example for one of the models:
```
python -m unittest tests.test_bm_models.TestCNN1D1L
```


## Contributing to ERT

Checklist for contributions:

1. Make sure that you updated the poetry environment (`poetry.lock`) and resolved all dependency conflicts. To update the environment:

```
poetry update
```

2. If functionality introduces a new dataset or model, design and implement tests to ensure the expected functionality with `unittest`.

3. Make sure that you run and pass all the tests:

```
poetry run python -m unittest
```

TBC with issues, pull requests, and branching guidelines. 

Also describe the github actions with CI.