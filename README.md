# SPHRED
A tensorflow re-implementation of SPHRED from paper A Variational Framework for Dialogue Generation, Shen et al.2017: https://arxiv.org/abs/1705.00316.

Reports, presentation, and demo of this project can be found [here](https://drive.google.com/drive/u/2/folders/1FmZ9qctybFdIb7Zr0BEoQvkDW1SY-rZ3?usp=sharing)

This project is originally forked from https://github.com/jshmSimon/VHRED.git, then modified to turn it into SPHRED

## Library specifications:
- tensorflow-addons version :   0.11.2
- tensorflow version        :   2.3.1
- gensim version            :   4.0.1

## Parameters:
Specified in configs.py, modify them by changing values of the args dictionary

## Run code:
- run /main.py
- Set train to True to run vhred_train model (Follow instructions in mains/vhred_train.py to train new model/ continue training model/ validate trained model)
- Set train to False to run the demo on trained model in model/ckpt/
