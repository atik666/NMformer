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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the parser
parser = argparse.ArgumentParser(
    description="NMformer base classifier."
    )

parser.add_argument('-p', '--path', type=str, default='./data/', help='Path to the dataset')
parser.add_argument('-model', '--model_path', type=str, default='./saved_models/', help='Path to saved model folder')
parser.add_argument('-test', '--test_db', type=list, default=['5.5dB', '7.5dB', '9.5dB'], help='Path to test images')
parser.add_argument('-b', '--batchsz', type=int, default=128, help='Batch size')
parser.add_argument('-sz', '--img_size', type=int, default=224, help='Image size')
parser.add_argument('-l', '--lr', type=float, default=0.00005, help='Learning rate')
parser.add_argument('-m_ft', '--model_ft', type=str, default='vit_ft.pth', help='Finetuned model file')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of  epochs')
parser.add_argument('-g', '--gpu', type=list, default=[0,1], help='GPU ids. Expand if necessary: [0,1,2,3,4,5]')

# Parse the arguments
args = parser.parse_args()
print(args)

# Load data 

# Define the paths to your training data folders

test_folders = [f for f in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, f)) and f in args.test_db]
test_paths = [os.path.join(args.path, f, 'test') for f in test_folders]

transform = T.Compose([T.Resize(args.img_size), 
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize images
                       ])

# Create a list of ImageFolder datasets for each folder
test_datasets = [ImageFolder(root=folder_path, transform=transform) for folder_path in test_paths]

# Concatenate the datasets into a single dataset
test_ds = ConcatDataset(test_datasets)

# Assuming train_ds is a ConcatDataset containing multiple datasets
classes = []
for dataset in test_ds.datasets:
    classes.extend(dataset.classes)   
classes = (list(set(classes)))
print(classes)

test_loader = DataLoader(test_ds, batch_size=args.batchsz, shuffle=True)

print("Number of test samples: ", len(test_ds))

vision_transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

vision_transformer.heads = nn.Linear(in_features=768, out_features=len(classes), bias=True)

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
print(train_on_gpu)

# move model to GPU
if train_on_gpu:
    vision_transformer.to('cuda')
    vision_transformer = torch.nn.DataParallel(vision_transformer, device_ids=args.gpu)

model_path = f'{args.model_path}{args.model_ft}'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    # Strip the "module." prefix from the keys if present
    state_dict = checkpoint['model_state_dict']
    # if all(key.startswith('module.') for key in state_dict.keys()):
    #     state_dict = {key[7:]: value for key, value in state_dict.items()}
    # # vision_transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    vision_transformer.load_state_dict(state_dict)
    print("Model loaded successfully!")

# specify loss function
criterion = nn.CrossEntropyLoss()

test_loss = 0.0
accuracy = 0

# number of classes
n_class = len(classes)

class_correct = np.zeros(n_class)
class_total = np.zeros(n_class)

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

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

    # Convert tensor to numpy array and append to lists
    true_labels.extend(targets.cpu().numpy())
    predicted_labels.extend(preds.cpu().numpy())
    
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
print('Test Accuracy of Dataset: \t {:.2f}% \t ({}/{})'.format(accuracy, 
                                                           int(np.sum(class_correct)), int(np.sum(class_total)) ))

# Calculate precision, recall, and F1 score
precision = []
recall = []
f1_score = []

for c in range(n_class):
    class_precision = class_correct[c] / (class_total[c] + 1e-9)
    class_recall = class_correct[c] / (targets.eq(c).sum().float() + 1e-9)
    class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-9)
    
    precision.append(class_precision.item())
    recall.append(class_recall.item())
    f1_score.append(class_f1.item())

# Overall precision, recall, and F1 score
overall_precision = sum(precision) / n_class
overall_recall = sum(recall) / n_class
overall_f1_score = sum(f1_score) / n_class

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("\nPrecision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()