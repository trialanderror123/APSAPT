import os
import pandas as pd
import numpy as np
import torch
from transformers import *
from src.config import *

def load_train_data():
    data = pd.read_csv(TRAIN_DATA_PATH)
    return data

def load_prediction_data():
    data = pd.read_csv(PREDICTION_DATA_PATH)
    return data

def device_loader():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def model_loader():
    # Load model, the pretrained model will include a single linear classification layer on top for classification. 
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABELS))
    if MODEL_PATH != "" and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    return model.cuda() if torch.cuda.is_available() else model

def optimizer_loader(model):
        # setting custom optimization parameters. You may implement a scheduler here as well.
    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in NO_DECAY)],
        'weight_decay_rate': WEIGHT_DECAY_RATE},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in NO_DECAY)],
        'weight_decay_rate': 0.0}
    ]

    return AdamW(optimizer_grouped_parameters,lr=LR,correct_bias=True)

def logFinder(value):
    return 1 if value == 0 else 1 + np.log(2 * value)