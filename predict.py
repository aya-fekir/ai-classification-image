import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import seaborn as sns
import argparse
import json



#arguments image prediction
parser = argparse.ArgumentParser(description='Image classifier prediction directory')

parser.add_argument('image_dir', help='image path directory', type=str)
parser.add_argument('check_dir', help='checkpoint directory', type=str)
parser.add_argument('--topk', help='top K most likely classes', type=int)
parser.add_argument('--category_names', help='mapping of categories to real names', type=str)
parser.add_argument('--gpu', help='turn on the GPU', type=str)

results = parser.parse_args()


#Model loading function
def load_checkpoint(filepath):
   
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


# Scales, crops, and normalizes a PIL image for a PyTorch model returns an Numpy array
def process_image(image):
    #Open the image
   
     #check the size of the image
    check_image = Image.open(image)
    
    #check the image size
    orig_width, orig_height = check_image.size

    #change the size in the aspect ratio
    if orig_width < orig_height:
        change_size=[256, 256**600]
    else:
        change_size=[256**600, 256]
    
    check_image.thumbnail(size = change_size)
    
    #crop the image accordingly 
    center = orig_width/4, orig_height/4

    left = center[0]-(224/2)
    upper = center[1]-(224/2)
    right = center[0]+(224/2)
    lower = center[1]+(224/2)
    
    check_image = check_image.crop((left, upper, right, lower))
    
    #Color channels as floats
    np_image = np.array(check_image)/255
    
    #Image normaliation
    norm_means = [0.485, 0.456, 0.406]
    norm_sd = [0.229, 0.224, 0.225]
    
    np_image = (np_image-norm_means)/norm_sd
    
    #Color channel as first dimension
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def predict(image_path, model, topk):
    
    model.to('cpu')
    model.eval();
    
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor)
    
    logps = model.forward(torch_image)

    linps = torch.exp(logps)
    
    top_probs, top_labels = linps.topk(topk)
    
    top_probs = top_probs.tolist()[0]
    top_labels = top_labels.tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, flowers


#File path argument
file_path = results.image_dir

#GPU if provided
if results.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

#Category name if provided, else, take the defult
if results.category_names:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#Loading the model from the checkpoint
model = load_checkpoint(results.check_dir)

#Number of TOP_K 
if results.topk:
    top_k = results.topk
else:
    top_k = 5

#Call the predict fuction and make predictions
top_probs, top_labels, classes = predict(file_path, model, top_k)


#Taking the class names from the classes
class_names = [cat_to_name [item] for item in classes]

#Print the results
for cl in range(len(class_names)):
     print("Probability level: {}/{}  ".format(cl+1, top_k),
            "Class name: {}   ".format(class_names [cl]),
            "Probability: {:.3f}% ".format(top_probs [cl]*100),
            )