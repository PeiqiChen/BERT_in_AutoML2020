import torch
import numpy as np
import random
from JigsawDataProcessor import JigsawDataProcessor
from BERT_Model import BERT_Model


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

data = JigsawDataProcessor()
train_dataloader, validation_dataloader =data.get_data_loaders()


model = BERT_Model(train_dataloader, validation_dataloader)
model.train()
model.save_model()





