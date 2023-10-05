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


### Folders: /datasets and /output configuration
By default, to facilitating the development of multiple components, docker-compose.yml is configured to map the dockers images folders 
`\datasets` `\outputs` and the file `configuration.json` to a single location in the repository root's directory. 

If you do not want a single `\datasets`, `\outputs` folder to all the docker images when running docker in development, 
eliminate the volumes mapping by commenting the lines in `docker-compose.yml` file. For example:

`"./datasets:/app/datasets"`

`"./outputs:/app/outputs"`

`"./configuration.json:/app/configuration.json"`

Then, the docker images will map the `/datasets`, `/outputs` and `configuration.json` file from the ones inside each component. 