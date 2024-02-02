# Pre Processing Bio-Measurements (BM) Modality

Pre-processing for BM Modality, collected using shimmer device.
It includes the data: ......

# Local Installation and run

## Set up virtual environment

`python -m venv ./venv`

`source ./venv/bin/activate`

## Install requirements

`pip install -r requirements.txt`

## Set up your local variables

1. Create your .env file

`cp example.env .env`

2. Change the variable values required to be changed according to the user's needs in the `.env` file, e.g., 
`PATH_CUSTOM_SETTINGS`, 
`DATASETS_FOLDER`, 
`OUTPUTS_FOLDER`

## Run the python script:

`python preprocess.py`

----

**Note**: participants data included in the initial implementation of BM preprocessing component:
P1 (second session), P3, P5, P6, P7, P8, P10, P11, P13
