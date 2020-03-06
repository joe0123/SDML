#SDML HW1
## Environment
* Linux
## Install
* tensorflow 2.0.0
* keras 2.3.0
* keras-bert
* numpy 1.17.2
* sklearn 0.21.3
* pandas  

If you want to run lib/pretrained/pretrain_bert.py  

* re
* math
* random

## Preparation
Because the limited upload size in Ceiba, we cannot attach our pretrained model. Please install [https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) and spread them into lib/pretrained first. Then pretrain the model before training.

## Pretraining
**`python pretrain_bert.py (pretrained_dir) (train) (test)`**  
ex. `python ./lib/pretrained/pretrain_bert.py ./lib/pretrained ./task2_trainset ./task2_public_testset`  
## Training
**`python fine_tuning.py (pretrained_dir) (train_set)`**  
ex. `python ./src/fine_tuning.py ./lib/pretrained ./task2_trainset`  
## Testing
**`python testing.py (pretrained_dir) (test_set) (model) (threshold 0) (threshold 1) (threshold 2) (threshold 3) > out.csv`**  
ex. `python ./src/testing.py ./lib/pretrained ./task2_public_testset ./src/weight.hdf5 0.39 0.24 0.24 0.45`  
=> This example can reproduce the result on Tbrain.  
