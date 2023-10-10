# SSL Features Extraction

Repository containing the components for training the models used in XR2Learn, including Encoders (SSL).


[Diagram with Architecture Overview](https://drive.google.com/file/d/1k3yLi9Y8tasFMJFNxIwKY-nRJzPdKPLw/view?usp=sharing)

### Installing 

Dependencies:
- Docker (Nvidia-Docker for CUDA)
- Python 3.10


## Docker Commands
Build a docker image

`docker compose build <service-name>`

Run a docker image

`docker compose run --rm <service-name>`

Run a docker image giving EnvVars

`KEY=VALUE docker compose run --rm <service-name>`

Run a docker image with shell entrypoint

`docker compose run --rm <service-name> \bin\bash`

Run all dockers from the Training Domain

`./run_all_dockers.sh`

#### Running on GPU 
docker compose -f docker-compose.yml -f docker-compose-gpu.yml run --rm <service-name>

### Folders: /datasets and /output configuration
By default, to facilitating the development of multiple components, docker-compose.yml is configured to map the dockers images folders 
`\datasets` `\outputs` and the file `configuration.json` to a single location in the repository root's directory. 

If you do not want a single `\datasets`, `\outputs` folder to all the docker images when running docker in development, 
eliminate the volumes mapping by commenting the lines in `docker-compose.yml` file. For example:

`"./datasets:/app/datasets"`

`"./outputs:/app/outputs"`

`"./configuration.json:/app/configuration.json"`

Then, the docker images will map the `/datasets`, `/outputs` and `configuration.json` file from the ones inside each component. 

### Configuration arguments
```
├── dataset_config: 
│   ├── dataset_name: name of dataset, should match folder name
│   ├── number_of_labels: number of labels in the dataset
├── pre_processing_config
│   ├── process: preprocessing to apply, standardize or normalize
│   ├── create_splits: create dataset splits
│   ├── target_sr: samplerate to resample to
│   ├── padding: add zero-padding or not
│   ├── max_length: desired maximum lenght
├── handcrafted_features_config
│   ├──
├── encoder_config
│   ├── from_module: module where the encoder is to be found
│   ├── class_name: name of encoder inside module
│   ├── input_type: specify the input modality
│   ├── kwargs: arguments for the encoder
├── ssl_config
│  ├── from_module: model from which to load the ssl framework
│  ├── ssl_framework: name of the framework to use
│  ├── epochs: number of epochs
│  ├── batch_size: batch size for SSL training
│  ├── kwargs: other arguments for SSL training
├── sup_config
│  ├── epochs: number of epochs to train for
│  ├── batch_size: batch size for supervised trianing
│  ├── use_augmentation_in_sup: weather to use the defined augmentations in sup learning or not
│  ├── kwargs: other supervised learning args
├── augmentations
│  ├── augmentation name
│     ├── probability: probability of augmentation to be applies
│     ├── kwargs: arguments for the augmentation to use
├── transforms:
│  ├── class_name: name of transform to apply
│  ├── from_module: where to fidn the transform
│  ├── transform_name: name of transformation
│  ├── in_test: if transformation is to be applied to test set or not   
```