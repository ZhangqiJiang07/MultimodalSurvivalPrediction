# Pancancer survival prediction using a deep learning architecture with multimodal representation and integration


## Data
*Please download the data from [Genomic Data Commons (GDC)](https://gdc.cancer.gov/about-data/publications/pancanatlas).* Make sure the original multimodal data is in the 'MultimodalSurvivalPrediction/preprocess/data' directory before you start running preprocessings and experiments.

## Experiments
If you’d like to run experiment with specified modality combination (e.g. only clinical and miRNA) use the `Clinical_xx_main.py` files.

To run the experiment of your choice, simply type:
```
>> cd <Path_to_MultimodalSurvivalPrediction_folder>
>> python3 Clinical_xx_main.py
```
