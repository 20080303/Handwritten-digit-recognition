# Handwritten-digit-recognition
Handwritten digit recognition
寄啊以为是28号交
还好可以套框架
面向百度谷歌编程开始了
裁缝怪 就硬整

首先是数据处理一波
康康训练集
csv不是很会处理 忙了很久
42000张图片 784像素
小看了一下差不多0-9每个数字都在4000左右

分别把标签和数据提出来
好像测试集没给标签
但是给了个输出示例
那就把训练集拆开
套一个sklearn

训练时间好久，下次如果有时间的话换个框架重新写一下
sklearn只能调用cpu
看了一下还只能用一核

第一次训练
训练时间:90.42522501945496s
准确率=0.9813333333333333
差不多吧就这样摆烂了
警告什么的忽略就好
应该没写错吧

今天的代码生涯就到这里了
大伙再见

