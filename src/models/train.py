import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from sklearn.metrics import f1_score, accuracy_score
from transformers import *
from tqdm import trange
from src.config import *
from src.utils import device_loader, model_loader, optimizer_loader
from src.data.dataloader import dataloader_train
from src.setup import setup

def train(model, optimizer, train_dataloader, device, num_labels=len(LABELS)):
    # Store our loss and accuracy for plotting
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(EPOCHS, desc="Epoch"):

        # Training
        
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0 #running loss
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_token_types = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # print(b_labels)
            # # Forward pass for multiclass classification
            # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # loss = outputs[0]
            # logits = outputs[1]

            # Forward pass for multilabel classification
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            loss_func = BCEWithLogitsLoss() 
            loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
            loss_func = BCELoss() 
            loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
            train_loss_set.append(loss.item())    

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

    return model

def validation(model, validation_dataloader, device):
     # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Variables to gather full output
    logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

    # Predict
    for i, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

    # Flatten outputs
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate Accuracy
    threshold = 0.5
    pred_bools = [pl>threshold for pl in pred_labels]
    true_bools = [tl==1 for tl in true_labels]
    val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
    val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

    print('F1 Validation Accuracy: ', val_f1_accuracy)
    print('Flat Validation Accuracy: ', val_flat_accuracy)

def main():
    setup()
    device = device_loader()

    train_dataloader, validation_dataloader = dataloader_train()

    model = model_loader()

    optimizer = optimizer_loader(model)

    model = train(model, optimizer, train_dataloader, device)
    validation(model, validation_dataloader, device)

    torch.save(model.state_dict(), 'bert_model')


if __name__ == '__main__':
    main()
