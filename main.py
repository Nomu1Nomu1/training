import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    # Auto detect
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATA_PATH = r"D:\code\kuliah\robot-ai\training\train"               # Folder path
    NUM_CLASSES = 2              # Banyak Folder/class di Path
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224             # Size gambar              
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0 # Buat GPU, 0 buat CPU

    # Data transform
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "train _data"), 
        transform=train_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Loaded {len(train_dataset)} training images")
    print(f"Classes: {train_dataset.classes}")


    # Tranfer Learning buat 1000 lebih images
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    model = model.to(device)

    # Freeze layer awal buat faster training
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': model.layer4.parameters(), 'lr': LEARNING_RATE*0.1},{'params': model.fc.parameters(), 'lr': LEARNING_RATE}], lr=LEARNING_RATE)   # Cuma buat optimize layer terakhir
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
    
        scheduler.step()
    
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    
        if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"resnet18_epoch_{epoch+1}.pth")
                print(f"Model saved at epoch {epoch+1}")
    
        print("\nTraining finished!")
        torch.save(model.state_dict(), "final_model.pth")
        print("Final model saved as final_model.pth")