import numpy as np
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils import data
from torch.autograd import Variable
from torchvision.io import read_image
from model import GTSRBnet
import utils 

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--verbose', type=bool, default=True, metavar='V',
                    help='verbose (default: True)')   
parser.add_argument('--save-model', type=bool, default=True, metavar='V',
                    help='For Saving the current Model') 
parser.add_argument(('--output-dir'), type=str, default='output', metavar='OP',
                    help='Output directory (default: output)')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
	use_gpu = False
	print("Using CPU")

# Create output directory if it does not exist
output_path = os.path.join(os.getcwd(), args.output_dir)
if not os.path.exists(output_path):
    os.makedirs(output_path)
# Create trained models directory if it does not exist
trained_models_path = os.path.join(output_path, 'trained_models')
if not os.path.exists(trained_models_path):
    os.makedirs(trained_models_path)

# Define path of training data
train_data_path = os.path.join(os.getcwd(), 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')
train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform=utils.transform)

# Divide data into training and validation set
train_ratio = 0.9
n_train_examples = int(len(train_data) * train_ratio)
n_val_examples = len(train_data) - n_train_examples
train_data, val_data = data.random_split(train_data, [n_train_examples, n_val_examples])
if args.verbose:
    print(f"Number of training samples = {len(train_data)}")
    print(f"Number of validation samples = {len(val_data)}")

# Get the number of classes and the class names
num_train_classes = len(train_data.dataset.classes)
train_hist = [0]*num_train_classes
for i in train_data.indices:
    tar = train_data.dataset.targets[i]
    train_hist[tar] += 1

num_val_classes = len(val_data.dataset.classes)
val_hist = [0]*num_val_classes
for i in val_data.indices:
    tar = val_data.dataset.targets[i]
    val_hist[tar] += 1

plt.figure(figsize=(10, 5))
plt.bar(range(num_train_classes), train_hist, label="train")
plt.bar(range(num_val_classes), val_hist, label="val")
#plt.bar(range(num_test_classes), test_hist, label="test")
legend = plt.legend(loc='upper right', shadow=True)
plt.title("Distribution Plot")
plt.xlabel("Class ID")
plt.ylabel("# of examples")
# Save the plot
plt.savefig(os.path.join(output_path, 'distribution.png'))

# Create data loader for training and validation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
val_loader = data.DataLoader(val_data, shuffle=True, batch_size=args.batch_size)

# Initialize the model and optimizer
model = GTSRBnet(num_train_classes)
model = model.to(device);

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)


def train():
    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)   
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim = 1)[1]
        correct += (max_index == target).sum()
        training_loss += loss
    avg_train_loss = training_loss / len(train_loader.dataset)
    avg_train_acc = 100. * correct / len(train_loader.dataset)
    return avg_train_loss.detach().cpu().numpy(), avg_train_acc.detach().cpu().numpy()

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss,2))
    validation_acc = 100. * correct / len(val_loader.dataset)
    return validation_loss, validation_acc

# Train the model
train_loss, train_acc = [], []
val_loss, val_acc = [], []
if args.verbose:
    print("Training started...")
for epoch in tqdm(range(1, args.epochs+1), disable=not(args.verbose)):
    avg_train_loss, avg_train_acc = train()
    avg_val_loss, avg_val_acc = validation()
    train_loss.append(avg_train_loss)
    train_acc.append(avg_train_acc)
    val_loss.append(avg_val_loss)
    val_acc.append(avg_val_acc)
    model_file = os.path.join(trained_models_path, 'model_' + str(epoch) + '.pth')
    torch.save(model.state_dict(), model_file)
if args.verbose:
    print("Training completed!")
    print("Model saved to {}".format(model_file))

# Plot training and validation loss
fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
ax[0].plot(train_loss, label='train', lw=2)
ax[0].plot(val_loss, label='val', lw=2)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
#ax[0].set_ylim([0, 0.1])
# remove right and top spines
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].plot(train_acc, label='train', lw=2)
ax[1].plot(val_acc, label='val', lw=2)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%)')
ax[1].legend()
# remove right and top spines
ax[1].spines['right'].set_visible(False)    
ax[1].spines['top'].set_visible(False)
#ax[1].set_ylim([95, 100])
# Save figure
fig.savefig(os.path.join(output_path, 'loss_acc.png'))
if args.verbose:
    print("Loss and accuracy plots saved to {}".format(os.path.join(output_path, 'loss_acc.png')))
