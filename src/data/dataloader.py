import torch
from src.features.build_features import build_features
from src.utils import load_train_data, load_prediction_data
from src.config import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
from sklearn.model_selection import train_test_split



def dataloader_train():
    data = load_train_data()
    data = build_features(data)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # tokenizer
    encodings = tokenizer.batch_encode_plus(list(data.content.values),max_length=MAX_LENGTH,pad_to_max_length=True) # tokenizer's encoding method
    print('tokenizer outputs: ', encodings.keys())

    input_ids = encodings['input_ids'] # tokenized and encoded sentences
    token_type_ids = encodings['token_type_ids'] # token type ids
    attention_masks = encodings['attention_mask'] # attention masks

    # Identifying indices of 'one_hot_labels' entries that only occur once - this will allow us to stratify split our training data later
    label_counts = data.one_hot_labels.astype(str).value_counts()
    one_freq = label_counts[label_counts==1].keys()
    one_freq_idxs = sorted(list(data[data.one_hot_labels.astype(str).isin(one_freq)].index), reverse=True)
    print('data label indices with only one instance: ', one_freq_idxs)

    # Gathering single instance inputs to force into the training set after stratified split
    one_freq_input_ids = [input_ids.pop(i) for i in one_freq_idxs]
    one_freq_token_types = [token_type_ids.pop(i) for i in one_freq_idxs]
    one_freq_attention_masks = [attention_masks.pop(i) for i in one_freq_idxs]
    one_freq_labels = [LABELS.pop(i) for i in one_freq_idxs]
    print("input_ids: {}, list(data.one_hot_labels.values): {}, token_type_ids: {},attention_masks: {}".format(len(input_ids), len(list(data.one_hot_labels.values)), len(token_type_ids),len(attention_masks)))

    train_inputs, validation_inputs, train_labels, validation_labels, train_token_types, validation_token_types, train_masks, validation_masks = train_test_split(input_ids, list(data.one_hot_labels.values), token_type_ids,attention_masks,
                                                            random_state=2020, test_size=TEST_SIZE, stratify = list(data.one_hot_labels.values))

    # Add one frequency data to train data
    train_inputs.extend(one_freq_input_ids)
    train_labels.extend(one_freq_labels)
    train_masks.extend(one_freq_attention_masks)
    train_token_types.extend(one_freq_token_types)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    train_token_types = torch.tensor(train_token_types)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)
    validation_token_types = torch.tensor(validation_token_types)

    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_types)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_types)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, validation_dataloader

def dataloader_prediction():
    data = load_prediction_data()
    data = data[data['content'].notna()]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # tokenizer
    encodings = tokenizer.batch_encode_plus(list(data.content.values),max_length=MAX_LENGTH,pad_to_max_length=True) # tokenizer's encoding method
    input_ids = torch.tensor(encodings['input_ids'], dtype=torch.int32) # tokenized and encoded sentences
    attention_masks = torch.tensor(encodings['attention_mask'], dtype=torch.int32) # attention masks
    return input_ids, attention_masks
    