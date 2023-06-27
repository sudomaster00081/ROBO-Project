import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5TokenizerFast
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class ChatbotDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        # Tokenize input and target texts
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).squeeze()
        target_ids = self.tokenizer.encode(target_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).squeeze()

        return input_ids, target_ids

# Define the training function
def train(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_dataloader)

# Set up the tokenizer and model
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Load and preprocess your chatbot dataset
with open('input.txt') as f:
    input_texts = [line.rstrip() for line in f]# List of input texts

with open('output.txt') as f:
    target_texts = [line.rstrip() for line in f]  # List of target texts
    

# Create train and validation datasets
train_dataset = ChatbotDataset(input_texts, target_texts, tokenizer)

# Set batch size and create data loaders
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device
model = model.to(device)

# Set optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    loss = train(model, train_dataloader, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss}")

# Save the fine-tuned model
model.save_pretrained("fine-tuned-chatbot-model")
tokenizer.save_pretrained("fine-tuned-chatbot-model")
