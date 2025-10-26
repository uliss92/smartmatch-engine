
import pandas as pd
from utils import upload_data, upload_directory
import argparse
import pandas as pd
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW
from azureml.core import Workspace, Datastore
import os
from torch.cuda.amp import autocast, GradScaler

import time




parser = argparse.ArgumentParser()

 
parser.add_argument('--root', type=str)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.root + "/fine_tuning/final.csv")

# ds_root = args.root

# golden_path = Path(ds_root) / 'fine_tuning/final.csv'

model_path = Path(args.root) / 'fine_tuning/fine_tuned' 
model_path.resolve().as_posix()






df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df['SortedTuple'] = list(zip(df['product'], df['neighbor']))
df['SortedTuple'] = df['SortedTuple'].apply(lambda x: tuple(sorted(x)))
df = df.drop_duplicates(subset='SortedTuple').reset_index(drop=True).drop(columns=['SortedTuple'])
df['match'] = df['match'].astype(int)



df, test_df = train_test_split(
    df, 
    test_size=0.1, 
    random_state=42, 
    shuffle=True, 
    stratify=df['match']  # Pass the column for stratification
)


epoch = 3
batch_size = 8

# Set device: If CUDA (GPU) is available, use it, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Prepare pairs
pairs = list(zip(df["product"], df["neighbor"]))
labels = df["match"].tolist()


# Load LaBSE tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")

# Tokenize text pairs
inputs = tokenizer(
    [f"{p[0]} [SEP] {p[1]}" for p in pairs],
    padding=True,
    truncation=True,
    return_tensors="pt", max_length=128
)

inputs = {key: value.to(device) for key, value in inputs.items()}
labels = torch.tensor(labels).float().to(device)



# Load pre-trained LaBSE model
base_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

# Add classification head
class MatchingModel(nn.Module):
    def __init__(self, base_model):
        super(MatchingModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, 1)  # Binary classification
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

model = MatchingModel(base_model).to(device)




# Custom dataset class
class MatchingDataset(Dataset):
    def __init__(self, inputs, labels):
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.labels = torch.tensor(labels).float()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

dataset = MatchingDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)


torch.cuda.empty_cache()
model.train()






scaler = GradScaler()

for epoch in range(epoch):  
    epoch_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch + 1}") as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.empty_cache()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Batch Loss": loss.item()})
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")



# for epoch in range(epoch):  # Number of epochs
#     epoch_loss = 0
#     with tqdm(dataloader, desc=f"Epoch {epoch + 1}") as pbar:
#         for batch in pbar:
#             # Transfer data to the selected device (e.g., GPU or CPU)
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
                    
#             # Forward pass
#             logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
#             loss = loss_fn(logits, labels)
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             torch.cuda.empty_cache()  # Clear cache after batch processing
            
#             epoch_loss += loss.item()
#             pbar.set_postfix({"Batch Loss": loss.item()})
#     print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")




# Save the model's state_dict
torch.save(model.state_dict(), "fine_tuned_model.pth")

# Save the tokenizer (same as before)
tokenizer.save_pretrained("fine_tuned_tokenizer")


cmd_ = f"cp -rf ./fine_tuned_model.pth {model_path} && cp -rf ./fine_tuned_tokenizer {model_path}"
os.system(cmd_)

print(model_path)
print(cmd_)





torch.cuda.empty_cache()











#test_df metrics

model.to(device)

model.eval()

# Prepare test pairs (assuming 'product' and 'neighbor' columns exist in test_df)
test_pairs = list(zip(test_df["product"], test_df["neighbor"]))



# Define test dataset
class TestDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_pair = f"{self.pairs[idx][0]} [SEP] {self.pairs[idx][1]}"
        inputs = self.tokenizer(
            text_pair, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }

# Prepare test dataset and dataloader
test_dataset = TestDataset(test_pairs, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Inference with batching
model.eval()
all_logits = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
        all_logits.append(logits.cpu())

# Convert collected results
all_logits = torch.cat(all_logits)
probabilities = torch.sigmoid(all_logits)
predictions = (probabilities > 0.5).int()



# # Tokenize the test pairs
# test_inputs = tokenizer(
#     [f"{p[0]} [SEP] {p[1]}" for p in test_pairs],
#     padding=True,
#     truncation=True,
#     return_tensors="pt", max_length=128
# )
# test_inputs = {key: value.to(device) for key, value in test_inputs.items()}






# # Make predictions on test data
# with torch.no_grad():
#     logits = model(input_ids=test_inputs["input_ids"], attention_mask=test_inputs["attention_mask"]).squeeze(-1)


torch.cuda.empty_cache()
# # Convert logits to probabilities (using sigmoid activation for binary classification)
# probabilities = torch.sigmoid(logits)

# # Convert probabilities to binary predictions (0 or 1)
# predictions = (probabilities > 0.5).int()

# Add predictions to the test dataframe
test_df['predicted_match'] = predictions.cpu().numpy()
test_df['score'] = probabilities.cpu().numpy()



y_true = test_df['match']
y_pred = test_df['predicted_match']

# Calculating the metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# Create a new DataFrame to store the metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC'],
    'Value': [accuracy, f1, precision, recall, roc_auc]
})

# Display the new DataFrame with metrics
metrics_df.to_csv('metrics_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)


time.sleep(2)  


# uploading
upload_data(
    datastore_name="mp_ds_matching",
    files=["metrics_df.csv", "test_df.csv" ],
    target_path="fine_tuning/"
)
