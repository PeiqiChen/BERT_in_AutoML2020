BERT模型实践代码，基于kaggle比赛，做了三个实验：

transformers-example（陈沛圻）：

- 使用原生 transformers 库提供的 BERT 自行训练， 3820210 行数据（数据集样例src/transformers-example/jigsaw-unintended-bias-in-toxicity-classification/train.csv），验证集准确度0.81

azure_autoML（许多）：使用 Azure automated machine learning 训练

- BERT 使用100 行数据（数据集样例src/azure_autoML/text/train.csv），验证集准确度0.808
- biLSTM 使用 1000 行数据（数据集样例src/azure_autoML/text/train.csv），验证集准确度0.731



