import os
import sys
sys.path.append(os.getcwd())
import torch
from transformers import *
import pandas as pd
import numpy as np
from src.config import *
from src.utils import device_loader, model_loader, logFinder
from src.data.dataloader import dataloader_prediction
from src.setup import setup

def predict(model, input_ids, attention_mask, device):
    pred_labels = []

    with torch.no_grad():
        outs = model(input_ids.to(device), attention_mask.to(device))
    b_logit_pred = outs[0]
    pred_label = torch.sigmoid(b_logit_pred)

    b_logit_pred = b_logit_pred.detach().cpu().numpy()
    pred_label = pred_label.to('cpu').numpy()

    pred_labels.append(pred_label)

    pred_labels = [item for sublist in pred_labels for item in sublist]
    pred_bools = [pl>=max(pl) for pl in pred_labels] #boolean output after thresholding
    
    labels = []
    for pred_bool in pred_bools:
        labels.append(LABELS[np.where(pred_bool == True)[0][0]])
    
    return labels

def weighted_sentiment_analysis(labels):
    df = pd.read_csv(PREDICTION_DATA_PATH)
    df['labels'] = labels
    reply = df['replyCount'].apply(lambda x: logFinder(x))
    retweet = df['retweetCount'].apply(lambda x: logFinder(x))
    likes = df['likeCount'].apply(lambda x: logFinder(x))
    df['weightedSentimentScore'] = 1 * reply * retweet * likes

    x, y = df.weightedSentimentScore.min(), df.weightedSentimentScore.max()
    df['weightedSentimentNorm'] = (df.weightedSentimentScore - x) / (y - x) * 1 + 0

    df.to_csv("yolo.csv", index = False)

def main():
    setup()
    device = device_loader()

    test_input_ids, test_attention_masks = dataloader_prediction()

    model = model_loader()

    labels = predict(model, test_input_ids, test_attention_masks, device)
    
    weighted_sentiment_analysis(labels)
    

if __name__ == "__main__":
    main()