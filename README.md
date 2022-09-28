# text-classifier-dp

文本分类，使用深度学习算法，如Linear、CNN、LSTM等

## 数据来源

使用`waimai_10k`数据集，该数据集主要是外卖餐饮的评论数据，包括4k条正向评论和8k条负向评论，本示例为使正负样本平衡，只选择4k条负向评论，共计8k条数据。

来源：[https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets)

### 运行

1. 运行`pretreatment.py`进行数据预处理；
2. 运行`vocabulary.py`构建词典，并将文本数据进行编码；
3. 运行`main.py`训练模型，并评估模型效果；
