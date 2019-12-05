# Cascading Convolutional Color Constancy

Huanglin Yu, Ke Chen*, Kaiqi Wang, Yanlin Qian, Zhaoxiang Zhang, Kui Jia &nbsp; &nbsp;
AAAI 2020

This implementation uses [Pytorch](http://pytorch.org/).

## Installation
Please install [Anaconda](https://www.anaconda.com/distribution/) firstly.

```shell
git clone https://github.com/yhlscut/C4.git
cd C4
## Create python env with relevant packages
conda create --name C4 python=3.6
conda activate C4
pip install -U pip
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch  # cudatoolkit=10.0 for cuda10
```

Tested on pytorch >= 1.0 and python3.

## Download
### Dataset

[*Shi's Re-processing of Gehler's Raw Dataset*:](http://www.cs.sfu.ca/~colour/data/shi_gehler/)

 - Download the 4 zip files from the website
 - Extract images in the `/cs/chroma/data/canon_dataset/586_dataset/png` directory into `./data/images/`, without creating subfolders.
 - Masking MCC chats: 
```shell
  bash ./data/run.sh
```

### Pretrained models
* Pretrained models can be downloaded [here](https://1drv.ms/u/s!AkGWFI5PP7sYarUAuXBGR3leujQ?e=Klqeg0). To reproduce the results reported in the paper, the pretrained models(*.pth) should be placed in `./trained_models/`, and then test model directly

## Run code

### Training
* Please train the three-fold models (modify `foldnum=0` to be `foldnum=1` or `foldnum=2` in line 6 of `train_sq_1stage.sh` and `train_sq_3stage.sh`)
* Train the C4_sq_1stage first:
```shell
bash ./scripts/train_sq_1stage.sh
```
* Train the C4_sq_3stage (Before that, please put the directory `./log/C4_sq_1stage` in `./trained_model` before):
```shell
bash ./scripts/train_sq_3stage.sh
```

### Testing

* After training, put the trained models in  `C4/trained_model/`, and run:
```shell
bash ./scripts/test_sq_3stage.sh
```
* To reproduce the results reported in the paper, put the pretrained models(*.pth) downloaded from [here](https://1drv.ms/u/s!AkGWFI5PP7sYarUAuXBGR3leujQ?e=Klqeg0) in `./trained_models/`, and then test model directly.