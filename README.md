# Cascading Convolutional Color Constancy

Huanglin Yu, Ke Chen*, Kaiqi Wang, Yanlin Qian, Zhaoxiang Zhang, Kui Jia &nbsp; &nbsp;
AAAI 2020 [[paper](https://arxiv.org/pdf/1912.11180.pdf)]

This implementation uses [Pytorch](http://pytorch.org/).

## Installation
Please install [Anaconda](https://www.anaconda.com/distribution/) firstly.

```shell
git clone https://github.com/yhlscut/C4.git
cd C4
## Create python env with relevant packages
conda create --name C4 python=3.6
source activate C4
pip install -U pip
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch  # cudatoolkit=10.0 for cuda10
```

Tested on pytorch >= 1.0 and python3.

## Download
### Dataset

[Shi's Re-processing of Gehler's Raw Dataset:](http://www.cs.sfu.ca/~colour/data/shi_gehler/)

 - Download the 4 zip files from the website and unzip them 
 - Extract images in the `/cs/chroma/data/canon_dataset/586_dataset/png` directory into `./data/images/`, without creating subfolders.
 - Masking MCC chats: 
```shell
  bash ./data/run.sh
```

### Pretrained models
* Pretrained models can be downloaded [here](https://1drv.ms/u/s!AiCnTyx4Rm42bk_D5EUMxMe3zk4?e=6ntqWg). To reproduce the results reported in the paper, the pretrained models(*.pth) should be placed in `./trained_models/`, and then test model directly

## Run code
Open the visdom service
```shell
python -m visdom.server -p 8008

```
### Training
* Please train the three-fold models (modify `foldnum=0` to be `foldnum=1` or `foldnum=2` in line 6 of `./scripts/train_sq_1stage.sh` and `./scripts/train_sq_3stage.sh` accordingly)
* Train the C4_sq_1stage first:
```shell
bash ./scripts/train_sq_1stage.sh
```
* Train the C4_sq_3stage (Before that, please move the directory `./log/C4_sq_1stage` to `./trained_model`):
```shell
bash ./scripts/train_sq_3stage.sh
```

### Testing

* After training, move the trained models directory in `./log/C4_sq_3stage` to `./trained_model/`, and run:
```shell
bash ./scripts/test_sq_3stage.sh
```
* To reproduce the results reported in the paper, move the pretrained models(*.pth) downloaded from [here](https://1drv.ms/u/s!AiCnTyx4Rm42bk_D5EUMxMe3zk4?e=6ntqWg) to `./trained_models/`, and then test model directly.

## Citing this work
If you find this code useful for your research, please consider citing the following paper:

	@article{yu2019cascading,
	  title={Cascading Convolutional Color Constancy},
	  author={Yu, Huanglin and Chen, Ke and Wang, Kaiqi and Qian, Yanlin and Zhang, Zhaoxiang and Jia, Kui},
	  journal={arXiv preprint arXiv:1912.11180},
	  year={2019}
	}

## Acknowledgements
This work is supported in part by the National Natural Science Foundation of China (Grant No.: 61771201,61902131), the Program for Guangdong Introducing Innovative and Enterpreneurial Teams (Grant No.:2017ZT07X183), the Fundamental Research Funds for the Central Universities (Grant No.: D2193130), and the SCUT Program (Grant No.: D6192110).