# 华为Ascend NPU训练的IMDB情感分类

## 简介

本项目为使用[mindspore](https://www.mindspore.cn/)实现的IMDB数据集情感分类任务。并使用SwanLab跟踪模型训练进展。

参考[MindSpore官方文档](https://www.mindspore.cn/tutorials/zh-CN/r2.4.1/nlp/sentiment_analysis.html#%E6%95%B0%E6%8D%AE%E9%9B%86%E9%A2%84%E5%A4%84%E7%90%86)，进行整理并简化了部分实现.

## 环境安装

由于华为Ascend环境安装较为复杂，建议参考[///](///)教程完成MindSpore环境安装

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

### 词编码器准备

使用如下命令下载+解压词编码器文件

```bash
wget -P ./embedding/ https://nlp.stanford.edu/data/glove.6B.zip
unzip embedding/glove.6B.zip -d embedding/
```

## 开始训练

使用如下命令开始训练

```
python train.py
```

可以看到