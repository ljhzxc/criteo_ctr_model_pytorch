# criteo_ctr_model_pytorch
用pytorch复现一下各种ctr模型，锻炼自己的动手能力，也加深自己对模型的理解。

criteo dataset: https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
该数据集解压后有两个文件，带label的train.txt(11G)，不带label的test.txt(1.4G)。为了快速试验，只选取了train.txt的前100w行数据(231M)。

## Reference
* 石晓文同学的tensorflow实战练习：https://github.com/princewen/tensorflow_practice
* 个人觉得写的挺好的pytorch ctr模型库，主要是FM类模型：https://github.com/rixwew/pytorch-fm
* 浅梦的DeepCTR Pytorch版：https://github.com/shenweichen/DeepCTR-Torch
