# 华为Ascend NPU训练的IMDB情感分类

## 简介

本项目为使用[mindspore](https://www.mindspore.cn/)实现的IMDB数据集情感分类任务。并使用SwanLab跟踪模型训练进展。

## 任务介绍

IMDB情感分类任务是一种自然语言处理任务，旨在分析IMDB（Internet Movie Database）电影评论中的文本内容，以判断评论的情感倾向，通常分为正面（Positive）和负面（Negative）两类。该任务广泛用于研究情感分析技术，尤其是在监督学习和深度学习领域。

数据集中通常包含预处理好的评论文本及其对应的情感标签，每条评论均标注为正面或负面。如下图：

![data_image](./docs/data_image.png)

LSTM（Long Short-Term Memory）是一种改进的循环神经网络，专为处理和预测序列数据中的长距离依赖而设计。与传统RNN相比，LSTM通过引入**记忆单元**和**门机制**，能够有效缓解梯度消失和梯度爆炸问题，使其在长序列数据的建模中表现优异。使用LSTM能轻松完成IMDB的语言情感分类任务。关于LSTM的具体原理建议参考[大神博客](https://blog.csdn.net/zhaojc1995/article/details/80572098)

![lstm](./docs/lstm.png)

本代码参考[MindSpore官方文档](https://www.mindspore.cn/tutorials/zh-CN/r2.4.1/nlp/sentiment_analysis.html#%E6%95%B0%E6%8D%AE%E9%9B%86%E9%A2%84%E5%A4%84%E7%90%86)，进行整理并简化了部分实现.


## 环境安装

### CPU环境安装

可以在CPU环境下安装MindSpore，虽然看起来没有Pytorch那么好用，但实际上文档还是写的很细的，真的很细，看得出华为工程师的严谨orz。配合sheng腾卡使用的话是非常有潜力的框架（MAC死活打不出sheng字）。

官方安装文档[link](https://www.mindspore.cn/install/)

也可以直接使用如下命令安装：

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.4.1/MindSpore/unified/x86_64/mindspore-2.4.1-cp311-cp311-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证安装成功命令：

```bash
python -c "import mindspore;mindspore.set_context(device_target='CPU');mindspore.run_check()"
```

如果输出如下信息说明MindSpore安装成功了：

```bash
MindSpore version: 2.4.1
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

### 华为Ascend NPU显卡环境安装

由于华为Ascend环境安装较为复杂，建议参考[MindSpore安装教程和踩坑记录](///)教程完成MindSpore环境安装

也附上官方安装教程链接[mindspore官方安装教程](https://www.mindspore.cn/install)，注意本教程使用的是[Mindspore 2.4.1](https://www.mindspore.cn/versions#2.4.1)，建议环境与本教程保持一致。

此外本教程使用[SwanLab](https://swanlab.cn)进行训练过程跟踪，SwanLab支持对Ascend系列NPU进行硬件识别和跟踪。

其他依赖环境安装方法：

```bash 
pip install -r requirements.txt
```

## 数据集&词编码文件准备

### 数据集准备

Linux使用如下命令完成下载+解压

```bash
wget -P ./data/ https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf data/aclImdb_v1.tar.gz -C data/
```

如果下载太慢可以使用[华为云提供的国内链接](https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz)下载。并且在`./data/`目录下解压。

> 如果解压不了tar.gz推荐安装[7zip解压器](https://www.7-zip.org/)，开源且通用的解压器

### 词编码器准备

使用如下命令下载+解压词编码器文件

```bash
wget -P ./embedding/ https://nlp.stanford.edu/data/glove.6B.zip
unzip embedding/glove.6B.zip -d embedding/
```

如果下载太慢可以使用[华为云提供的国内链接](https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/glove.6B.zip)下载。并且在`./embedding/`目录下解压。

## 开始训练

使用如下命令开始训练

```
python train.py
```

可是这

> 如果提示登录swanlab，可以参考[如何登录SwanLab](https://docs.swanlab.cn/guide_cloud/general/quick-start.html#_2-%E7%99%BB%E5%BD%95%E8%B4%A6%E5%8F%B7)，这样将能够使用**云上看版**随时查看训练过程与结果。

完成设置便可以在云上实时看到训练进展，我的实验记录可参考[完整实验记录](https://swanlab.cn/@ShaohonChen/Ascend_IMDB_CLS/charts)

![log_img](docs/log_img.png)

并且附上其他脚本与在线实验记录：

| 内容  | 训练命令  | 实验log  |
|--------|--------|--------|
| 基线 | `python train.py configs/baseline.json` | [log](https://swanlab.cn/@ShaohonChen/Ascend_IMDB_CLS/runs/bbuwb3291pxkoyi00zu16/chart) |
| CPU运行 | `python train.py configs/baseline.json CPU` | [log](https://swanlab.cn/@ShaohonChen/Ascend_IMDB_CLS/runs/htlcflrzpozfcds5o9q93/chart) |
| 双层LSTM | `python train.py configs/two_layer.json` | [log](https://swanlab.cn/@ShaohonChen/Ascend_IMDB_CLS/runs/pywgvlga53wozdb9c2toz/chart) |
| 小batch数 | `python train.py configs/small_batch.json` | [log](https://swanlab.cn/@ShaohonChen/Ascend_IMDB_CLS/runs/pywgvlga53wozdb9c2toz/chart) |
| 隐藏层加大 | `python train.py configs/large_hs.json` | [log](https://swanlab.cn/@ShaohonChen/Ascend_IMDB_CLS/runs/cx0implu0xoxffi57173c/chart) |
| 学习率加大 | `python train.py configs/large_hs.json` | [log](https://swanlab.cn/@ShaohonChen/Ascend_IMDB_CLS/runs/tsyxib42islmlsay1ogna/chart) |

相关超参数和最终结果可在[图标视图查看](https://swanlab.cn/@ShaohonChen/Ascend_IMDB_CLS/overview)

![log_table](docs/log_table.png)

> PS: 观察了下日志，发现还是训练量不足，应该增大些训练量（40-50epoch比较合适）