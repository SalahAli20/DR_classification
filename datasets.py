import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils import resample

import numpy as np 
import pandas as pd 
import os
from PIL import Image
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit


class MessidorDataset(Dataset):
    """
    Custom Dataset class for loading image data from a CSV file.
    """
    def __init__(self, csv_path, root_dir, image_size=256, transform=None, upsample=False):
        """
        Args:
            csv_path (str): Path to the CSV file containing image data.
            root_dir (str): Root directory containing the image files.
            image_size (integer): Dimension of the resized images.
            transform (torchvision.transforms, optional): Transformations to apply to images. Defaults to None.
            upsample (bool, optional): Whether to upsample minority classes to balance the dataset. Defaults to False.
        """
        self.data = pd.read_csv(csv_path)
        self.image_ids = self.data["id_code"].tolist()
        self.labels = self.data["diagnosis"].tolist()
        self.transform = transform
        self.root_dir = root_dir  
        self.image_size = image_size  
        self.upsample = upsample

        if self.upsample:
            self.upsample_data()

    def __len__(self):
        """
        Returns the length of the dataset (number of data points).
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns a data point at the specified index with its weight.
        Args:
            idx (int): Index of the data point.
        Returns:
            tuple: A tuple containing the image, its corresponding label
        """
        image_id = self.image_ids[idx]
        filepath = os.path.join(self.root_dir, image_id) 

        try:
            with Image.open(filepath) as img:
                image = img.resize((256,256))
                image = image.convert("RGB")  # Convert to RGB if needed
                image = np.array(image)
                image = np.transpose(image, (2, 0, 1))
                image = torch.tensor(image, dtype=torch.float32) / 255.0
        except FileNotFoundError:
            print(f"Error: Image file not found: {filepath}")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def upsample_data(self):
        """
        Performs upsampling to balance the dataset.
        """
        # Find the class with the maximum number of samples
        max_samples = max(self.labels.count(label) for label in set(self.labels))

        # Upsample minority classes
        for label in set(self.labels):
            class_count = self.labels.count(label)
            if class_count < max_samples:
                # Upsample minority class
                minority_data = [self.image_ids[i] for i, l in enumerate(self.labels) if l == label]
                minority_labels = [label] * class_count
                majority_data = [self.image_ids[i] for i, l in enumerate(self.labels) if l != label]

                upsampled_minority_data = resample(minority_data, replace=True, n_samples=max_samples - class_count, random_state=42)

                self.image_ids.extend(upsampled_minority_data)
                self.labels.extend([label] * (max_samples - class_count))

        combined = list(zip(self.image_ids, self.labels))
        np.random.shuffle(combined)
        self.image_ids, self.labels = zip(*combined)


def create_dataloaders(csv_path, root_dir, batch_size, output, test_size=0.2, val_size=0.1, transform=None):
    """
    Creates dataloaders for training, testing, and validation sets with optional class weights.
    Args:
        csv_path (str): Path to the CSV file containing image data.
        root_dir (str): Root directory containing the image files.
        batch_size (int): Batch size for dataloaders.
        test_size (float, optional): Proportion of data for the testing set. Defaults to 0.2.
        val_size (float, optional): Proportion of data for the validation set. Defaults to 0.1.
        transform (torchvision.transforms, optional): Transformations to apply to images. Defaults to None.
    Returns:
        tuple: A tuple containing three dataloaders for training, testing, and validation sets.
    """
    dataset = MessidorDataset(csv_path, root_dir, transform=transform, upsample=False)

    X_train, X_test, y_train, y_test = train_test_split(dataset.image_ids, dataset.labels, test_size=test_size, stratify=dataset.labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=42)

    training_df = pd.DataFrame({"id_code": X_train, "diagnosis": y_train}).to_csv('{}/training.csv'.format(output))
    testing_df = pd.DataFrame({"id_code": X_test, "diagnosis": y_test}).to_csv('{}/testing.csv'.format(output))
    val_df = pd.DataFrame({"id_code": X_val, "diagnosis": y_val}).to_csv('{}/validation.csv'.format(output))
    
    train_subset = MessidorDataset('{}/training.csv'.format(output), root_dir, transform=transform,upsample=True)
    test_subset = MessidorDataset('{}/testing.csv'.format(output), root_dir, transform=None,upsample=False)
    val_subset = MessidorDataset('{}/validation.csv'.format(output), root_dir, transform=None,upsample=False)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
