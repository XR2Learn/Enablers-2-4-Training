# SSL Features Extraction

Repository containing the components for training the models used in XR2Learn, including Encoders (SSL).

[Diagram with Architecture Overview](https://drive.google.com/file/d/1k3yLi9Y8tasFMJFNxIwKY-nRJzPdKPLw/view?usp=sharing)

### Installing 

Dependencies:
- Docker (Nvidia-Docker for CUDA)
- Python 3.10

### Quick-start
`python train.py <arg-list>`

## Docker Commands
Build a docker image

`docker compose build <service-name>`

Run a docker image

`docker compose run --rm <service-name>`

Run a docker image giving EnvVars

`KEY=VALUE docker compose run --rm <service-name>`