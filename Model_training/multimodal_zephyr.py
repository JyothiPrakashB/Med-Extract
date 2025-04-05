import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from multimodal_zephyr import MultimodalLLM  # Make sure this file is in the same directory

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "HuggingFaceH4/zephyr-7b-alpha"
img_path = "path/to/images"
batch_size = 2
num_epochs = 3
lr = 5e-5
max_length = 128

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Image processor
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom Dataset
class MultiModalDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_file = os.path.join(self.img_dir, row['new_relative_path'])
        image = Image.open(img_file).convert("RGB")
        image = image_transform(image)

        prompt = row['Hinglish_Question']
        answer = row['Hinglish_Answer']
        input_text = f"<image>\n{prompt}"
        output_text = answer

        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        outputs = tokenizer(output_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)

        return {
            "pixel_values": image,
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": outputs.input_ids.squeeze(0),
        }

# Load datasets
train_ds = MultiModalDataset("train.csv", img_path)
val_ds = MultiModalDataset("val.csv", img_path)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Model
model = MultimodalLLM().to(device)
model.train()

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        images = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save checkpoint
torch.save(model.state_dict(), "checkpoint_zephyr.pt")


n_epochs = 30
 
patience = 10
model, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)
