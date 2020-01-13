import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from datetime import datetime
import os
import glob
import copy
import sys
from collections import OrderedDict

#arguments for model training
parser = argparse.ArgumentParser(description='Image classifier training directory')

parser.add_argument('data_dir', help="data directory", type=str)
parser.add_argument('--save_dir', help='directory to save checkpoints', type=str)
parser.add_argument('--arch', help='vgg13 architecture', type=str)
parser.add_argument('--learning_rate', help='set learning rate as 0.01', type=float)
parser.add_argument('--hidden_units', help='hidden units as 512', type=int)
parser.add_argument('--epochs', help='number of epochs as 20', type=int)
parser.add_argument('--gpu', help='turn on the GPU', type=str)

results = parser.parse_args()

data_dir = results.data_dir 
train_dir = results.data_dir + '/train'
valid_dir = results.data_dir + '/valid'
test_dir = results.data_dir + '/test'



# Check if the GPU is availble, otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#transform the data
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                  }

#Load the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder('flowers'+'/train', transform=data_transforms['train']),
                  'valid': datasets.ImageFolder('flowers'+'/valid', transform=data_transforms['valid']),
                  'test': datasets.ImageFolder('flowers'+'/test', transform=data_transforms['test'])
                 } 
# Using the image datasets and the trainforms, define the dataloaders
dataloaders  = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
                 'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
               }

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
class_to_idx = image_datasets['train'].class_to_idx
idx_to_name = {
    idx: cat_to_name[key] for key, idx in image_datasets['train'].class_to_idx.items()
}
list(idx_to_name.items())[:10]


#Build and train  network  

model = models.vgg16(pretrained=True) 
 # check if GPU is availbl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device.type 
  # Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088,4096)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


model.classifier = classifier


     
 #defining our  criterion and optimizer 
criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
 #Transfering the model to the device
model.to(device)
    #training
epochs=13
steps=0
print_freq=30
print('epochs:', epochs, ', print_freq:', print_freq, ', lr:',0.001)
for epoch in range (epochs):
    running_loss = 0
    for inputs, labels in dataloaders['train']:
        steps+=1
        inputs,labels = inputs.to(device), labels.to(device)
            #clear gradient 
        optimizer.zero_grad()            
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_freq==0:
            valid_loss=0
            accuracy=0
            model.eval()
            with torch.no_grad():
                for inputs , labels in dataloaders['valid']:
                    inputs , labels = inputs.to(device),labels.to(device)
                    output=model.forward(inputs)
                    valid_loss+=criterion(output,labels).item()
                    ps=torch.exp(output)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()
            print(f"Epoch: {epoch} "
                 f"Training_loss: {running_loss/print_freq:.3f} "
                 f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f} "
                 f"Validation Accuracy: {accuracy/len(dataloaders['valid'])*100:.3f} "
                             )    
                
            running_loss=0
            model.train()
print("classifier training done ")



 # TODO: Do validation on the test setrunning_loss=0
test_loss = 0
accuracy = 0

with torch.no_grad():
    for images, labels in dataloaders['test']:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
               
        #Accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
print(f"Test accuracy: {accuracy/len(dataloaders['test'])*100:.3f}%")

def load_checkpoint(filepath):
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False
        
    checkpoint = torch.load(filepath)
    
    #Extract classifier
    model.classifier = checkpoint['classifier']
    model.cat_to_name = checkpoint['cat_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']
    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    if(train_on_gpu):
        model = model.to('cuda')
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    total_param = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')
    print(f'Model has been trained for {model.epochs} epochs.')
    

    return model, optimizer
