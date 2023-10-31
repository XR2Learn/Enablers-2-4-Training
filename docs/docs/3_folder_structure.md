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