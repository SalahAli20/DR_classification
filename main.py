import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import pandas as pd
from models import ModelFactory
from utils import train_model,plot_metrics,plot_loss
from datasets import create_dataloaders
from config import ModelConfig

import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DR classification models')    
    parser.add_argument('--input_CSV', help='path to csv file')
    parser.add_argument('--input_folder', help='path to input data')
    parser.add_argument('--output_folder', help='path to output data')
    parser.add_argument('--config_file',default='config.json', help='path to config_file')
    parser.add_argument('--model', default='DenseNet', choices=['VGG19', 'DenseNet', 'MyModel'], help='name of model to use')
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                      help="Specify 'train' for training or 'test' for model evaluation.")
    args = parser.parse_args()
    

    
    # csv_path ='/kaggle/input/messidor2preprocess/messidor_data.csv'
    # root_dir = '/kaggle/input/messidor2preprocess/messidor-2/messidor-2/preprocess'

    csv_path=args.input_CSV
    root_dir=args.input_folder
    output_folder=args.output_folder
    config_file=args.config_file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('input CSV is {}'.format(csv_path))
    print('input image dir  is {}'.format(root_dir))
    print('output folder is {}'.format(output_folder))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    model_config = ModelConfig(config_file)
   
    if args.mode == "train":
        train_loader, test_loader, val_loader = create_dataloaders(csv_path,root_dir, output=output_folder,batch_size=model_config.batch_size,transform=None)
        train_data=pd.read_csv('{}/training.csv'.format(output_folder)) 
        model = ModelFactory.create_model(args.model, num_classes=5, num_layers_to_retrain=model_config.trainable_layers)
        optimizer = torch.optim.SGD(model.parameters(), lr= model_config.learning_rate)    
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        trained_model, losses, epochs, accuracies, precisions, recalls, f1_scores=train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs= model_config.epochs, device=device)
        torch.save(trained_model.state_dict(),'{}/model_weights.pth'.format(output_folder))

        plot_metrics(epochs,accuracies, precisions, recalls, f1_scores,'{}/val_metrics.png'.format(output_folder))
        plot_loss((epochs, losses,'{}/loss.png'.format(output_folder)))

    elif args.mode == "test":
         overall_accuracy, precision, recall, f1=inference(model,csv_file,root_dir)
        print(f'Accuracy: {overall_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

       
