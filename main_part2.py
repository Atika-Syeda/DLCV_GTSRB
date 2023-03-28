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
from model import GTSRBnet
import utils 

# TODO: Add reference: https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
# TODO: Add reference: https://arxiv.org/pdf/1710.05381.pdf
# TODO: Decide how to select a factor for the weighted sampler

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--verbose', type=bool, default=True, metavar='V',
                    help='verbose (default: True)')   
parser.add_argument('--output-dir', type=str, default='output_part2', metavar='OP',
                    help='Output directory (default: output_part2)')
parser.add_argument('--sampler', type=str, default='weighted', metavar='S',
                    help='Sampler (default: weighted)')
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
trained_models_path = os.path.join(output_path, args.sampler+'_trained_models')
if not os.path.exists(trained_models_path):
    os.makedirs(trained_models_path)

# Define path of training data
train_data_path = os.path.join(os.getcwd(), 'GTSRB/Final_Training/Images')
dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform=utils.transform)

# Divide data into training and validation set
train_ratio = 0.9
n_train_examples = int(len(dataset) * train_ratio)
n_val_examples = len(dataset) - n_train_examples
train_data, val_data = data.random_split(dataset, [n_train_examples, n_val_examples])
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

y_train_indices = train_data.indices
y_train = [dataset.targets[i] for i in y_train_indices]
class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
# Find weights for each class
weight = 1. / class_sample_count

# Create a weighted sampler
if args.sampler == 'weighted':
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
elif args.sampler == 'oversample': 
    # Oversample the minority classes by a factor of 10. This increases the number of cases in the minority classes so that the number matches the majority classes.
    minority_classes = np.where(class_sample_count < 1000)[0]
    samples_weight = np.array([weight[t] for t in y_train])
    for i in minority_classes:
        samples_weight[y_train == i] *= 10
    samples_weight = torch.from_numpy(samples_weight)
elif args.sampler == 'undersample': 
    # Undersample the majority classes by a factor of 10. Undersampling decreases the number of cases in the majority classes to match the minority classes. 
    majority_classes = np.where(class_sample_count > 1000)[0]
    samples_weight = np.array([weight[t] for t in y_train])
    for i in majority_classes:
        samples_weight[y_train == i] /= 10
    samples_weight = torch.from_numpy(samples_weight)
elif args.sampler == 'both':
    # Oversample the minority classes by a factor of 10 and undersample the majority classes by a factor of 10. 
    minority_classes = np.where(class_sample_count < 1000)[0]
    majority_classes = np.where(class_sample_count > 1000)[0]
    samples_weight = np.array([weight[t] for t in y_train])
    for i in minority_classes:
        samples_weight[y_train == i] *= 10
    for i in majority_classes:
        samples_weight[y_train == i] /= 10
    samples_weight = torch.from_numpy(samples_weight)
else:
    samples_weight = np.array([1 for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)

# Plot the histogram of the training data classes 
plt.figure(figsize=(10, 5))
plt.hist(samples_weight, bins=100)
plt.title("Samples weights")
plt.xlabel("Weight")
plt.ylabel("Number of samples")
# Save the plot
plt.savefig(os.path.join(output_path, args.sampler+'_sample_weights.png'))


# Create data loader for training and validation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
train_loader = data.DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
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
# Save figure
fig.savefig(os.path.join(output_path, args.sampler+'_loss_acc.png'))
if args.verbose:
    print("Loss and accuracy plots saved to {}".format(os.path.join(output_path, args.sampler+'_loss_acc.png')))

