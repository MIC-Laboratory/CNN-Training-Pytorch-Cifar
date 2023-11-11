import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import sys
import yaml
import argparse
import deepspeed
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy
from tqdm import tqdm
from Models.Resnet import ResNet101
from Models.Mobilenetv2 import MobileNetV2
from Models.Vgg import VGG
from Weapon.WarmUpLR import WarmUpLR

from deepspeed.accelerator import get_accelerator
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.join(os.getcwd()))
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3,OFAResNets


# argument for deepspeed
def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=128,
                        type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('-e',
                        '--epochs',
                        default=200,
                        type=int,
                        help='number of total epochs (default: 200)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")

    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        type=int,
                        nargs='+',
                        default=[
                            1,
                        ],
                        help='number of experts list, MoE related.')
    parser.add_argument(
        '--mlp-type',
        type=str,
        default='standard',
        help=
        'Only applicable when num-experts > 1, accepts [standard, residual]')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )
    parser.add_argument(
        '--dtype',
        default='fp32',
        type=str,
        choices=['bf16', 'fp16', 'fp32'],
        help=
        'Datatype used for training'
    )
    parser.add_argument(
        '--stage',
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help=
        'Datatype used for training'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

# Init deepspeed
deepspeed.init_distributed()

# Read the configuration from the config.yaml file
with open("CNN_Training_Pytorch_Cifar/config.yaml","r") as f:
    config = yaml.load(f,yaml.FullLoader)["Training_seting"]
# Read deepspeed config
with open("CNN_Training_Pytorch_Cifar/ds_config.yaml","r") as f:
    ds_config = yaml.load(f,yaml.FullLoader)


# Read the configuration from the config.yaml file
batch_size = config["batch_size"]
training_epoch = config["training_epoch"]
num_workers = config["num_workers"]
lr_rate = config["learning_rate"]
warmup_epoch = config["warmup_epoch"]
best_acc = 0


# Set the dataset mean, standard deviation, and input size based on the chosen dataset
if config["dataset"] == "Cifar10":
    dataset_mean = [0.4914, 0.4822, 0.4465]
    dataset_std = [0.2470, 0.2435, 0.2616]
    input_size = 32
elif config["dataset"] == "Cifar100":
    dataset_mean = [0.5071, 0.4867, 0.4408]
    dataset_std = [0.2675, 0.2565, 0.2761]
    input_size = 32




# Set the paths for dataset, weights, models, and log data
dataset_path = config["dataset_path"]
weight_path = os.path.join(config["weight_path"],config["dataset"],config["models"]) 
filename = config["models"]
log_path = os.path.join(config["experiment_data_path"],config["dataset"],config["models"]) 
writer = SummaryWriter(log_dir=log_path)


# Set the paths for dataset, weights, models, and log data
train_transform = transforms.Compose(
    [
    transforms.RandomCrop(input_size,padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.autoaugment.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.Normalize(mean=dataset_mean,std=dataset_std)
    ])
test_transform = transforms.Compose([
    transforms.RandomCrop(input_size,padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean,std=dataset_std)
])

if torch.distributed.get_rank() != 0:
    # might be downloading cifar data, let rank 0 download first
    torch.distributed.barrier()

# Load the dataset based on the chosen dataset (Cifar10, Cifar100, or Imagenet) and apply the defined transformations
print("==> Preparing data")
if config["dataset"] == "Cifar10":
    train_set = torchvision.datasets.CIFAR10(dataset_path,train=True,transform=train_transform,download=True)
    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()
    test_set = torchvision.datasets.CIFAR10(dataset_path,train=False,transform=test_transform,download=True)
elif config["dataset"] == "Cifar100":
    train_set = torchvision.datasets.CIFAR100(dataset_path,train=True,transform=train_transform,download=True)
    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()
    test_set = torchvision.datasets.CIFAR100(dataset_path,train=False,transform=test_transform,download=True)

# Get the number of classes in the dataset
classes = len(train_set.classes)


# Create data loaders for testing
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

# Create an instance of the selected model (ResNet101, MobileNetV2, or VGG) and transfer it to the chosen device
print("==> Preparing models")
print(f"==> Using deepspeed mode")
if config["models"] == "ResNet101":
    net = ResNet101(num_classes=classes)
if config["models"] == "ResNet-OFA":
    net = OFAResNets(
        n_classes=classes,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.1,
        depth_list=4,
        expand_ratio_list=6,
        width_mult_list=1.0, 
    )
elif config["models"] == "Mobilenetv2":
    net = MobileNetV2(num_classes=classes)
elif config["models"] == "VGG16":
    net = VGG(num_class=classes)

args = add_argument()

def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }

    return split_params_into_different_moe_groups_for_optimizer(parameters)


parameters = filter(lambda p: p.requires_grad, net.parameters())
if args.moe_param_group:
    parameters = create_moe_param_groups(net)

# Define the loss function and optimizer for training the model
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


model_engine, optimizer, train_loader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=train_set, config=ds_config)

local_device = get_accelerator().device_name(model_engine.local_rank)
local_rank = model_engine.local_rank

# For float32, target_dtype will be None so no datatype conversion needed
target_dtype = None
if model_engine.bfloat16_enabled():
    target_dtype=torch.bfloat16
elif model_engine.fp16_enabled():
    target_dtype=torch.half


# Validation function
def validation(network,dataloader,file_name,save=True):
    # Iterate over the data loader
    global best_acc
    accuracy = 0
    running_loss = 0.0
    total = 0
    correct = 0
    network.eval()
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)


                # Perform forward pass and calculate loss and accuracy
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                pbar.update()
                pbar.set_description_str("Acc: {:.3f} {}/{} | Loss: {:.3f}".format(accuracy,correct,total,running_loss/(i+1)))
            
            # Save the model's checkpoint if accuracy improved
            if not os.path.isdir(weight_path):
                os.makedirs(weight_path)
            check_point_path = os.path.join(weight_path,"Checkpoint.pt")
            torch.save({"state_dict":network.state_dict(),"optimizer":optimizer.state_dict()},check_point_path)    
            if accuracy > best_acc:
                best_acc = accuracy
                if save:
                    PATH = os.path.join(weight_path,f"Model@{config['models']}_ACC@{best_acc}.pt")
                    torch.save({"state_dict":network.state_dict()}, PATH)
                    print("Save: Acc "+str(best_acc))
                else:
                    print("Best: Acc "+str(best_acc))
    return running_loss/len(dataloader),accuracy


# Training function
def train(epoch,network,dataloader):
    # loop over the dataset multiple times
    running_loss = 0.0
    total = 0
    correct = 0
    network.train()
    with tqdm(total=len(dataloader)) as pbar:
        # Iterate over the data loader
        for i, data in enumerate(dataloader, 0):
            
            inputs, labels = data[0].to(local_device), data[1].to(local_device)
            if target_dtype != None:
                inputs = inputs.to(target_dtype)


            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            network.backward(loss)
            network.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()            
            
            
            accuracy = 100 * correct / total
            pbar.update()
            pbar.set_description_str("Epoch: {} | Acc: {:.3f} {}/{} | Loss: {:.3f}".format(epoch,accuracy,correct,total,running_loss/(i+1)))


# Training and Testing Loop
print("==> Start training/testing")
for epoch in range(training_epoch + warmup_epoch):
    train(epoch, network=model_engine,dataloader=train_loader)
    loss,accuracy = validation(network=net,file_name=filename,dataloader=test_loader)
    writer.add_scalar('Test/Loss', loss, epoch)
    writer.add_scalar('Test/ACC', accuracy, epoch)
writer.close()
print("==> Finish")