### Structure
The structure is made in such a way that each module/project has its own folder containing their code,requirements,... but share the dataset/outputs folders.

In the datasets folder each subfolder should be named with the datasets name. One level deeper each folder should represent a subject and it's contents the data recordings.

Within the outputs folders the preprocessed data/modalities shall be represented inside of a folder with the name of the modality and each file will contain identifiers to the dataset, subject and original file.
each file should be in _.npy_ format.

Within the test/train/val csvs no modality must be specified, only the file name, during training/inference the specified modality should match the modality folder name.

```
.
├── datasets/
│   ├── datasetName1/
│   │   ├── Subject1/
│   │   │   ├── filename1
│   │   │   └── filename2
│   │   ├── Subject2/
│   │   │   ├── filename1
│   │   │   └── filename2
│   │   └── ...
│   ├── datasetName2/
│   │   ├── Subject1
│   │   ├── Subject2
│   │   └── ...
│   └── ...
├── outputs/
│   ├── modality1/
│   │   ├── {datasetName1}_{Subject1}_{filename1}.npy
│   │   ├── ...
│   │   └── {datasetName1}_{Subject2}_{filename2}.npy
│   ├── modality2/
│   │   ├── {datasetName1}_{Subject1}_{filename1}.npy
│   │   ├── ...
│   │   └── {datasetName1}_{Subject2}_{filename2}.npy
│   ├── ssl_training/
│   │   ├── logs/
│   │   │   └── ...
│   │   └── encoder.pt
│   ├── supervised_training/
│   │   ├── logs/
│   │   │   └── ...
│   │   ├── classifier.pt
│   │   └── model.pt
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── Handcrafted_Features_Extraction/
│   └── ...
├── Pre_processing/
│   └── ...
├── SSL_Features_Extraction/
│   └── ...
├── SSL_Training/
│   └── ...
└── Supervised_Training/
    └── ...
```

### Folders configuration: /datasets and /output 

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