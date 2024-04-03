import os
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def inference(model,csv_file,root_dir):
    test_subset = MessidorDataset(csv_path, root_dir, transform=None)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    test_preds = []
    test_targets = []  
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs}, Validation', leave=False)
        for inputs, labels in test_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    cm = confusion_matrix(test_targets, test_preds)
    class_accuracies = np.nan_to_num(cm.diagonal() / cm.sum(axis=1), nan=0.0)
    overall_accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, average='weighted')
    recall = recall_score(test_targets, test_preds, average='weighted')
    f1 = f1_score(test_targets, test_preds, average='weighted')


    return overall_accuracy, precision, recall, f1, cm

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device=torch.device('cuda')):
    model.to(device)
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    epochs = []

    for epoch in range(num_epochs):
        # Training phase with progress bar
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}, Training', leave=False)
        model.train()  # Set model to training mode
        for inputs, labels in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_progress_bar.set_postfix({'loss': running_loss / len(train_loader.dataset)})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)

        # Evaluation phase on validation set with progress bar
        val_preds = []
        val_targets = []
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs}, Validation', leave=False)
            for inputs, labels in val_progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                     
        cm = confusion_matrix(val_targets, val_preds)
        class_accuracies = np.nan_to_num(cm.diagonal() / cm.sum(axis=1), nan=0.0)
        overall_accuracy = accuracy_score(val_targets, val_preds)
        precision = precision_score(val_targets, val_preds, average='weighted')
        recall = recall_score(val_targets, val_preds, average='weighted')
        f1 = f1_score(val_targets, val_preds, average='weighted')
        
        accuracies.append(overall_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        epochs.append(epoch + 1)

        # Print epoch loss and evaluation metrics
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {overall_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        
    return model, losses, epochs, accuracies, precisions, recalls, f1_scores

def plot_metrics(epochs,losses,accuracies, precisions, recalls, f1_scores,output):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Validation Metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='val Accuracy', color='green')
    plt.plot(epochs, precisions, label='val Precision', color='red')
    plt.plot(epochs, recalls, label='val Recall', color='orange')
    plt.plot(epochs, f1_scores, label='val  F1 Score', color='purple')
    plt.title('Validation Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(output)


def plot_loss(epochs, losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output)

