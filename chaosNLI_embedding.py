import sys
import os
import pickle 
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import src.data_utils as d

dataset = sys.argv[1]

df = d.df_creator(d.load_raw(dataset))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-base")
model = AutoModel.from_pretrained("cross-encoder/nli-deberta-base")

# Concatenate premise and hypothesis
# For some models, you might need to manually add special tokens like [SEP]
# But for many models, the tokenizer handles this automatically
combined_input = tokenizer(df["premise"].tolist(), df["hypothesis"].tolist(), padding=True, truncation=True, return_tensors="pt")

# Get the last hidden states
with torch.no_grad():
    outputs = model(**combined_input, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]

#mean pooling over the sequence_length dimension
last_hidden_states_mean = np.mean(last_hidden_states.numpy(), axis=1)

data = {
    "embedding":last_hidden_states_mean, 
    "premise": df["premise"].values, 
    "hypothesis": df["hypothesis"].values, 
    "label_dist": np.array(df["label_dist"].to_list()), 
    }


os.makedirs(os.path.join("data", "chaosNLI","embeddings"), exist_ok=True)      
with open(os.path.join("data", "chaosNLI", "embeddings",  dataset+".pkl"), 'wb') as f:
    pickle.dump(data, f)