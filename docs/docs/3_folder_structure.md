### Structure
The structure is made in such a
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
│   │   ├── {datasetName1}_{Subjec1}_{filename1}.npy
│   │   ├── ...
│   │   └── {datasetName1}_{Subjec2}_{filename2}.npy
│   ├── modality2/
│   │   ├── {datasetName1}_{Subjec1}_{filename1}.npy
│   │   ├── ...
│   │   └── {datasetName1}_{Subjec2}_{filename2}.npy
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