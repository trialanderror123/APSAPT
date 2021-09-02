from src.config import LABELS

def build_features(data):
    # Content is the coloumn name with processed text
    data = data[data['content'].notna()]
    for column in LABELS:
        data = data[data[column].notna()]
    data = fix_dtype(data)
    data['one_hot_labels'] = list(data[LABELS].values)
    processed_data = data[["content", "one_hot_labels"]]
    processed_data.reset_index(drop = True, inplace = True)
    return processed_data

def fix_dtype(data):
    data = data.astype({coloumn : int for coloumn in LABELS})
    return data