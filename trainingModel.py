import os
import shutil
from sklearn.model_selection import train_test_split

# Base dataset folder
base_dir = "C:/Users/Aadya/Desktop/FineTuneML/dataset"

# Paths to your images
labubu_dir = os.path.join(base_dir, "labubu_images")
lafufu_dir = os.path.join(base_dir, "lafufu_images")

# Get all image paths
labubu_images = [os.path.join(labubu_dir, f)
                 for f in os.listdir(labubu_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

lafufu_images = [os.path.join(lafufu_dir, f)
                 for f in os.listdir(lafufu_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

print(f"Found {len(labubu_images)} labubu images")
print(f"Found {len(lafufu_images)} lafufu images")

# Split into train/validation (80/20)
labubu_train, labubu_val = train_test_split(labubu_images, test_size=0.2, random_state=42)
lafufu_train, lafufu_val = train_test_split(lafufu_images, test_size=0.2, random_state=42)

# Create new directory structure
train_labubu_dir = os.path.join(base_dir, "train", "labubu")
train_lafufu_dir = os.path.join(base_dir, "train", "lafufu")
val_labubu_dir = os.path.join(base_dir, "validation", "labubu")
val_lafufu_dir = os.path.join(base_dir, "validation", "lafufu")

os.makedirs(train_labubu_dir, exist_ok=True)
os.makedirs(train_lafufu_dir, exist_ok=True)
os.makedirs(val_labubu_dir, exist_ok=True)
os.makedirs(val_lafufu_dir, exist_ok=True)

# Copy files to train
for img in labubu_train:
    shutil.copy(img, train_labubu_dir)
for img in lafufu_train:
    shutil.copy(img, train_lafufu_dir)

# Copy files to validation
for img in labubu_val:
    shutil.copy(img, val_labubu_dir)
for img in lafufu_val:
    shutil.copy(img, val_lafufu_dir)

print(f"\nTrain set - Labubu: {len(labubu_train)}, Lafufu: {len(lafufu_train)}")
print(f"Validation set - Labubu: {len(labubu_val)}, Lafufu: {len(lafufu_val)}")
print("\nDataset structure created successfully!")

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

print("GPU available:", torch.cuda.is_available())

# Custom Dataset class
class LabubuDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.processor = processor
        self.images = []
        self.labels = []
        
        # Load labubu images (label 0)
        labubu_dir = os.path.join(root_dir, "labubu")
        for img_name in os.listdir(labubu_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(labubu_dir, img_name))
                self.labels.append(0)
        
        # Load lafufu images (label 1)
        lafufu_dir = os.path.join(root_dir, "lafufu")
        for img_name in os.listdir(lafufu_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(lafufu_dir, img_name))
                self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()
        
        return {'pixel_values': pixel_values, 'labels': torch.tensor(label)}

# Load processor and model
print("Loading model...")
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=2,
    id2label={0: 'labubu', 1: 'lafufu'},
    label2id={'labubu': 0, 'lafufu': 1},
    ignore_mismatched_sizes=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create datasets
print("Loading datasets...")
train_dataset = LabubuDataset('dataset/train', processor)
val_dataset = LabubuDataset('dataset/validation', processor)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

print("\nStarting training...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            val_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    print(f'\nEpoch {epoch+1} Summary:')
    print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

# Save model
print("Saving model...")
model.save_pretrained("./labubu_model_final")
processor.save_pretrained("./labubu_model_final")
print("Training complete!")
