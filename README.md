# Pancancer survival prediction using a deep learning architecture with multimodal representation and integration


## Data
Please download the data from [Genomic Data Commons (GDC)](https://gdc.cancer.gov/about-data/publications/pancanatlas) and [University of California Santa Cruz (UCSC) Xena](http://xena.ucsc.edu/public/). Make sure the original multimodal datasets are in the 'MultimodalSurvivalPrediction/preprocess/data' directory before you start running preprocessings and experiments.

Or you can download the preprocessed data from our [Google Drive](https://drive.google.com/file/d/1Cu1hVO_kPQGpeGHxgYHzLVDH7ulqF8tN/view?usp=share_link) and make sure the unzipped multimodal datasets are in the 'MultimodalSurvivalPrediction/preprocess/preprocessed_data' directory.


## Install packages
```
pip install -r requirements.txt
```

## Experiments
I.
If you’d like to run experiment with specified modality combination (e.g. clinical, miRNA, and mRNA) for pancancer survival prediction, you first need to modify the 'modalities_list' variable in 'pancancer_prediction.py'.
```
modalities_list = [['clinical', 'mRNA', 'miRNA']]
```
Then, to run the experiment of your choice, simply type:
```
>> cd <Path_to_MultimodalSurvivalPrediction_folder>
>> python3 pancancer_prediction.py
```

II.
If you’d like to run experiment with specified modality combination (e.g. clinical, miRNA, and mRNA) with pancancer training dataset for single cancer survival prediction, you first need to modify the 'modalities_list' variable in 'single_cancer_prediction.py'.
```
modalities_list = [['clinical', 'mRNA', 'miRNA']]
```
And then type:
```
>> cd <Path_to_MultimodalSurvivalPrediction_folder>
>> python3 single_cancer_prediction.py
```
