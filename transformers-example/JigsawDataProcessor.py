import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split


# 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
batch_size = 32
output_dir = './model_save/'
model_name='bert-base-uncased'




class JigsawDataProcessor:
    def __init__(self):
        self.train_data_file = "jigsaw-unintended-bias-in-toxicity-classification/train.csv"
        self.test_data_file = "jigsaw-unintended-bias-in-toxicity-classification/test.csv"
        self.tokenizer=BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.chunk_size = 256
        self.maxlen = 512

    def get_data(self):
        df = pd.read_csv(self.train_data_file, sep=',', header=0,low_memory=False,
                         names=['id','target','comment_text','severe_toxicity',
                                'obscene','identity_attack','insult','threat','asian',
                                'atheist','bisexual','black','buddhist','christian',
                                'female','heterosexual','hindu','homosexual_gay_or_lesbian',
                                'intellectual_or_learning_disability','jewish',
                                'latino','male','muslim','other_disability','other_gender',
                                'other_race_or_ethnicity','other_religion','other_sexual_orientation',
                                'physical_disability','psychiatric_or_mental_illness',
                                'transgender','white','created_date','publication_id',
                                'parent_id','article_id','rating','funny','wow','sad',
                                'likes','disagree','sexual_explicit','identity_annotator_count',
                                'toxicity_annotator_count'])

        # 打印数据集的记录数
        # print(df)
        print('Number of training sentences: {:,}\n'.format(df.shape[0]))
        # df.loc[df.label == 0].sample(5)[['target', 'comment_text']]

        comment_text = df.comment_text.values
        labels = df.target.values
        bilabels = []
        for label in labels:
            if label!= 0.0:
                bilabels.append(1)
            else:
                bilabels.append(0)
        # lenth = 1000
        # return comment_text[:lenth], bilabels[:lenth]
        return comment_text, bilabels

    '''使用 BERT 的 tokenizer编码每一条输入文本'''
    def encode(self):
        """
        Encoder for encoding the text into sequence of integers for BERT Input
        """
        print("begin encoding")
        input_ids = []
        attention_masks = []
        comment_text, labels = self.get_data()
        print("done gettinng data")

        for sent in comment_text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=64,  # 填充 & 截断长度
                pad_to_max_length=True,
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                truncation=True
            )

            # 将编码后的文本加入到列表
            input_ids.append(encoded_dict['input_ids'])

            # 将文本的 attention mask 也加入到 attention_masks 列表
            attention_masks.append(encoded_dict['attention_mask'])
        print("get inputisd and masks")

        # 将列表转换为 tensor
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        print("all to tensor")


        # 输出第 1 行文本的原始和编码后的信息
        # print('Original: ', comment_text[0])
        # print('Token IDs:', input_ids[0])
        self.tokenizer.save_pretrained(output_dir)
        print("done enncoding")


        return input_ids, attention_masks,labels

    '''用编码后的数据构建训练集和测试集'''
    def get_data_loaders(self):
        print("begin get_data_loaders")

        input_ids, attention_masks, labels = self.encode()
        from torch.utils.data import TensorDataset, random_split
        # 将输入数据合并为 TensorDataset 对象
        dataset = TensorDataset(input_ids, attention_masks, labels)
        # 计算训练集和验证集大小
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # 按照数据大小随机拆分训练集和测试集
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print("begin get_data_loaders -train")

        # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
        train_dataloader = DataLoader(
            train_dataset,  # 训练样本
            sampler=RandomSampler(train_dataset),  # 随机小批量
            batch_size=batch_size  # 以小批量进行训练
        )
        print("begin get_data_loaders -validation")

        # 验证集不需要随机化，这里顺序读取就好
        validation_dataloader = DataLoader(
            val_dataset,  # 验证样本
            sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
            batch_size=batch_size
        )
        print("done get_data_loaders")

        return train_dataloader, validation_dataloader

