import torchvision.models as models
import torch.optim as optim
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.parallel
import os
import argparse
from tqdm import tqdm # type: ignore

# Initialize the parser
parser = argparse.ArgumentParser(
    description="NMformer base classifier."
    )

parser.add_argument('-p', '--path', type=str, default='./data/', help='Path to the dataset')
parser.add_argument('-model', '--model_path', type=str, default='./saved_models/', help='Path to saved model folder')
parser.add_argument('-train', '--train_db', type=list, default=['5.5dB', '7.5dB', '9.5dB'], help='Path to train images')
parser.add_argument('-test', '--test_db', type=list, default=['5.5dB', '7.5dB', '9.5dB'], help='Path to test images')
parser.add_argument('-b', '--batchsz', type=int, default=128, help='Batch size')
parser.add_argument('-sz', '--img_size', type=int, default=224, help='Image size')
parser.add_argument('-l', '--lr', type=float, default=0.00005, help='Learning rate')
parser.add_argument('-m', '--model', type=str, default='vit.pth', help='Main model file')
parser.add_argument('-p', '--opt', type=str, default='optim.pth', help='Main optimizer file')
parser.add_argument('-m_ft', '--model_ft', type=str, default='vit_ft.pth', help='Finetuned model file')
parser.add_argument('-p_ft', '--opt_ft', type=str, default='optim_ft.pth', help='Finetuned optimizer file')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of  epochs')
parser.add_argument('-g', '--gpu', type=list, default=[0,1], help='GPU ids. Expand if necessary: [0,1,2,3,4,5]')

# Parse the arguments
args = parser.parse_args()
print(args)

# Load data 

# Define the paths to your training data folders

train_folders = [f for f in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, f)) and f in args.train_db]
test_folders = [f for f in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, f)) and f in args.test_db]

train_paths = [os.path.join(args.path, f, 'train') for f in train_folders]
test_paths = [os.path.join(args.path, f, 'test') for f in test_folders]

transform = T.Compose([T.Resize(args.img_size), 
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize images
                       ])

# Create a list of ImageFolder datasets for each folder
train_datasets = [ImageFolder(root=folder_path, transform=transform) for folder_path in train_paths]
test_datasets = [ImageFolder(root=folder_path, transform=transform) for folder_path in test_paths]

# Concatenate the datasets into a single dataset
train_ds = ConcatDataset(train_datasets)
test_ds = ConcatDataset(test_datasets)

# Assuming train_ds is a ConcatDataset containing multiple datasets
classes = []
for dataset in train_ds.datasets:
    classes.extend(dataset.classes)   
classes = (list(set(classes)))
print("Number of classes:", classes)

train_loader = DataLoader(train_ds, batch_size=args.batchsz, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args.batchsz, shuffle=True)

print("Number of train samples: ", len(train_ds))
print("Number of test samples: ", len(test_ds))

vision_transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
print(vision_transformer)

print(vision_transformer.heads)

# fine-tune with dataset

# change the number of output classes
vision_transformer.heads = nn.Linear(in_features=768, out_features=len(classes), bias=True)

# freeze the parameters except the last linear layer

# freeze weights
for p in vision_transformer.parameters():
    p.requires_grad = False # TODO: make true for training from scratch

# unfreeze weights of classification head to train
for p in vision_transformer.heads.parameters():
    p.requires_grad = True

# check whether corresponding layers are frozen

for layer_name, p in vision_transformer.named_parameters():
    print('Layer Name: {}, Frozen: {}'.format(layer_name, not p.requires_grad))
    print()

# Check for a GPU
# move model to GPU
if torch.cuda.is_available():
    vision_transformer.to('cuda')
    vision_transformer = torch.nn.DataParallel(vision_transformer, device_ids=args.gpu)

# specify loss function
criterion = nn.CrossEntropyLoss()

# define optimizer
# only train the parameters with requires_grad set to True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, vision_transformer.parameters()), lr=args.lr)

# load model if it exists

model_path = f'{args.model_path}{args.model}'
optim_path = f'{args.model_path}{args.opt}'

save_model_path = f'{args.model_path}{args.model_ft}'
save_optim_path = f'{args.model_path}{args.model_ft}'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    # Strip the "module." prefix from the keys if present
    state_dict = checkpoint['model_state_dict']
    # if all(key.startswith('module.') for key in state_dict.keys()):
    #     state_dict = {key[7:]: value for key, value in state_dict.items()}
    # # vision_transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    vision_transformer.load_state_dict(state_dict)
    print("Model loaded successfully!")

if os.path.exists(optim_path):
    optimizer.load_state_dict(torch.load(optim_path))
    print("Optimizer loaded successfully!")

# Train model 

train_loss_list, valid_loss_list = [], []

# prepare model for training
vision_transformer.train()

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    valid_loss = checkpoint['valid_loss']
    best_valid_loss = valid_loss
    valid_accuracy = checkpoint['valid_accuracy']
    best_valid_accuracy = valid_accuracy
else:
    best_valid_loss = float('inf')  # Initialize with a very large value
    best_valid_accuracy = 0.0

for e in tqdm(range(args.epochs), desc="Epochs"):
    train_loss = 0.0
    valid_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # get batch data
    for i, (images, targets) in enumerate(tqdm(train_loader, desc="Batch")):
        
        # move to gpu if available
        if torch.cuda.is_available():
            images, targets = images.to('cuda'), targets.to('cuda')
        
        # clear grad
        optimizer.zero_grad()
        
        # feedforward data
        outputs = vision_transformer(images)
        
        # calculate loss
        loss = criterion(outputs, targets)
        
        # backward pass, calculate gradients
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # track loss
        train_loss += loss.item()
    
    # set model to evaluation mode
    vision_transformer.eval()
    
    # validate model
    for images, targets in test_loader:
        
        # move to gpu if available
        if torch.cuda.is_available():
            images = images.to('cuda')
            targets = targets.to('cuda')
        
        # turn off gradients
        with torch.no_grad():
            
            outputs = vision_transformer(images)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()

            # calculate validation accuracy
            _, predicted = torch.max(outputs, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
    # set model back to trianing mode
    vision_transformer.train()
    
    # get average loss values
    train_loss = train_loss / len(train_loader)
    valid_loss = valid_loss / len(test_loader)
    valid_accuracy = correct_predictions / total_predictions

    if valid_loss < best_valid_loss or valid_accuracy > best_valid_accuracy:
        best_valid_loss = valid_loss
        best_valid_accuracy = valid_accuracy
        # Save model parameters along with validation loss information
        save_dict = {
            'model_state_dict': vision_transformer.state_dict(),
            'valid_loss': valid_loss,
            'valid_accuracy': valid_accuracy
        }
        torch.save(save_dict, save_model_path) 
        torch.save(optimizer.state_dict(), save_optim_path)
        print("Best model saved! \n")
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    
    # output training statistics for epoch
    print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f} \t Validation Accuracy: {:.2f}%'
                  .format( (e+1), train_loss, valid_loss, valid_accuracy * 100))
    

# visualize loss statistics

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# plot losses
x = list(range(1, args.epochs + 1))
plt.plot(x, train_loss_list, color ="blue", label='Train')
plt.plot(x, valid_loss_list, color="orange", label='Validation')
plt.legend(loc="upper right")
plt.xticks(range(1, args.epochs + 1, 10))

plt.savefig(f'{args.model_path}loss_plot_ft.png')

if os.path.exists(save_model_path):
    checkpoint = torch.load(save_model_path)
    # Strip the "module." prefix from the keys if present
    state_dict = checkpoint['model_state_dict']
    # if all(key.startswith('module.') for key in state_dict.keys()):
    #     state_dict = {key[7:]: value for key, value in state_dict.items()}
    # # vision_transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    vision_transformer.load_state_dict(state_dict)
    print("Model loaded successfully! \n")
else:
    # prepare model for evaluation
    vision_transformer.eval()

test_loss = 0.0
accuracy = 0

# number of classes
n_class = len(classes)

class_correct = np.zeros(n_class)
class_total = np.zeros(n_class)

# move model back to cpu
vision_transformer = vision_transformer.to('cuda')

# test model
for images, targets in test_loader:

    images, targets = images.to('cuda'), targets.to('cuda')
    
    # get outputs
    # outputs = vision_transformer(images)
    outputs = vision_transformer(images)
    
    # calculate loss
    loss = criterion(outputs, targets)
    
    # track loss
    test_loss += loss.item()
    
    # get predictions from probabilities
    preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
    
    # get correct predictions
    correct_preds = (preds == targets).type(torch.cuda.FloatTensor)
    
    # calculate and accumulate accuracy
    accuracy += torch.mean(correct_preds).item() * 100
    
    # calculate test accuracy for each class
    for c in range(n_class):
        
        class_total[c] += (targets == c).sum()
        class_correct[c] += ((correct_preds) * (targets == c)).sum()

# get average accuracy
accuracy = accuracy / len(test_loader)

# get average loss 
test_loss = test_loss / len(test_loader)

# output test loss statistics 
print('Test Loss: {:.6f}'.format(test_loss))

class_accuracy = class_correct / class_total

print('Test Accuracy of Classes')
print()

for c in range(n_class):
    print('{}\t: {}% \t ({}/{})'.format(classes[c], 
                                int(class_accuracy[c] * 100), int(class_correct[c]), int(class_total[c])) )

print()
print('Test Accuracy of Dataset: \t {:.2f}}% \t ({}/{})'.format(int(accuracy), 
                                                           int(np.sum(class_correct)), int(np.sum(class_total)) ))

