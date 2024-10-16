''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

from __future__ import print_function

import argparse

import torch
import torch.utils.data as td
import wandb
from torchvision import transforms

import dataprocessor
import experiment as ex
import model
import trainer
import utils

# parser = argparse.ArgumentParser(description='Recursive-CNNs')
# parser.add_argument('--batch-size', type=int, default=32, metavar='N',
#                     help='input batch size for training (default: 32)')
# parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
#                     help='learning rate (default: 0.005)')
# parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 30],
#                     help='Decrease learning rate at these epochs.')
# parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
#                     help='LR is multiplied by gamma[k] on schedule[k], number of gammas should be equal to schedule')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                     help='SGD momentum (default: 0.9)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--pretrain', action='store_true', default=False,
#                     help='Pretrain the model on CIFAR dataset?')
# parser.add_argument('--load-ram', action='store_true', default=False,
#                     help='Load data in ram:  : Remove this')
#
# parser.add_argument('--debug', action='store_true', default=True,
#                     help='Debug messages')
# parser.add_argument('--seed', type=int, default=2323,
#                     help='Seeds values to be used')
# parser.add_argument('--log-interval', type=int, default=5, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--model-type', default="resnet",
#                     help='model type to be used. Example : resnet32, resnet20, densenet, test')
# parser.add_argument('--name', default="noname",
#                     help='Name of the experiment')
# parser.add_argument('--output-dir', default="../",
#                     help='Directory to store the results; a new folder "DDMMYYYY" will be created '
#                          'in the specified directory to save the results.')
# parser.add_argument('--decay', type=float, default=0.00001, help='Weight decay (L2 penalty).')
# parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for trianing')
# parser.add_argument('--dataset', default="document", help='Dataset to be used; example document, corner')
# parser.add_argument('--loader', default="hdd",
#                     help='Loader to load data; hdd for reading from the hdd and ram for loading all data in the memory')
# parser.add_argument("-i", "--data-dirs", nargs='+', default="/Users/khurramjaved96/documentTest64",
#                     help="input Directory of train data")
# parser.add_argument("-v", "--validation-dirs", nargs='+', default="/Users/khurramjaved96/documentTest64",
#                     help="input Directory of val data")

experiment_names = "Experiment-10"

output_dir = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\experiments"
no_cuda = True
dataset_type = "complete_document"
data_dirs = [
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\augmentations",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\kosmos",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\self-collected",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\smart-doc"
]

validation_dirs = [
        r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\augmentations",
        r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\kosmos",
        r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\self-collected",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\smart-doc"
]
testing_dirs = [
        r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\augmentations",
        r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\kosmos",
        r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\self-collected",
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\smart-doc"
]
loader = "hdd_complete_doc"

model_type = "resnet"

pretrain = False

lr = 0.005
batch_size = 15
seed = 42

momentum = 0.9
decay = 0.00001

gammas = [.2, .2, .2, .2,.2,.2,.2,.2]

epochs = 75
schedule = [10, 20, 30, 40, 45,50,60,70]
cuda = not no_cuda and torch.cuda.is_available()

arguments = {
    "experiment_names": experiment_names,
    "output_dir": output_dir,
    "cuda": cuda,
    "data_dirs": data_dirs,
    "dataset_type": dataset_type,
    "validation_dirs": validation_dirs,
    "loader": loader,
    "model_type": model_type,
    "pretrain": pretrain,
    "lr": lr,
    "batch_size": batch_size,
    "seed": seed,
    "momentum": momentum,
    "decay": decay,
    "gammas": gammas,
    "epochs": epochs,
    "schedule": schedule,
}
#
wandb.init(project="full-document-detection",
           entity="kosmos-randd",
           config=arguments)

wandb.run.name = wandb.run.name + "-" + experiment_names
# Define an experiment.
my_experiment = ex.experiment(wandb.run.name, arguments, output_dir)
print(arguments)
# Add logging support
logger = utils.utils.setup_logger(my_experiment.path)
# %%
dataset = dataprocessor.DatasetFactory.get_dataset(data_dirs, dataset_type, "train1.csv")
# %%
dataset_val = dataprocessor.DatasetFactory.get_dataset(validation_dirs, dataset_type, "val1.csv")
# %%
dataset_test = dataprocessor.DatasetFactory.get_dataset(validation_dirs, dataset_type, "test1.csv")
# %%
# Fix the seed.
# seed = seed
# torch.manual_seed(seed)
# if cuda:
#     torch.cuda.manual_seed(seed)

train_transform = transforms.Compose([transforms.Resize([64, 64]),
                                      # transforms.ColorJitter(1.5, 1.5, 0.9, 0.5),
                                      transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize([64, 64]),
                                     transforms.ToTensor()])

train_dataset_loader = dataprocessor.LoaderFactory.get_loader(loader, dataset.myData,
                                                              transform=test_transform,
                                                              cuda=cuda)
# Loader used for training data
val_dataset_loader = dataprocessor.LoaderFactory.get_loader(loader, dataset_val.myData,
                                                            transform=test_transform,
                                                            cuda=cuda)  # Loader used for training data
test_dataset_loader = dataprocessor.LoaderFactory.get_loader(loader, dataset_test.myData,
                                                             transform=test_transform,
                                                             cuda=cuda)
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Iterator to iterate over training data.
train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                             batch_size=batch_size, shuffle=True, **kwargs)
# Iterator to iterate over training data.
val_iterator = torch.utils.data.DataLoader(val_dataset_loader,
                                           batch_size=batch_size, shuffle=True, **kwargs)
# Iterator to iterate over training data.
test_iterator = torch.utils.data.DataLoader(test_dataset_loader,
                                            batch_size=batch_size, shuffle=True, **kwargs)

# Get the required model
myModel = model.ModelFactory.get_model(model_type, "complete_document_64")
if cuda:
    myModel.cuda()

# Should I pretrain the model on CIFAR?
if pretrain:
    trainset = dataprocessor.DatasetFactory.get_dataset(None, "CIFAR")
    train_iterator_cifar = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    # Define the optimizer used in the experiment
    cifar_optimizer = torch.optim.SGD(myModel.parameters(), lr, momentum=momentum,
                                      weight_decay=decay, nesterov=True)

    # Trainer object used for training
    cifar_trainer = trainer.CIFARTrainer(train_iterator_cifar, myModel, cuda, cifar_optimizer)

    for epoch in range(0, 70):
        logger.info("Epoch : %d", epoch)
        cifar_trainer.update_lr(epoch, schedule, gammas)
        cifar_trainer.train(epoch)

    # Freeze the model
    counter = 0
    for experiment_names, param in myModel.named_parameters():
        # Getting the length of total layers so I can freeze x% of layers
        gen_len = sum(1 for _ in myModel.parameters())
        if counter < int(gen_len * 0.5):
            param.requires_grad = False
            logger.warning(experiment_names)
        else:
            logger.info(experiment_names)
        counter += 1

# Define the optimizer used in the experiment
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, myModel.parameters()), lr,
                            momentum=momentum,
                            weight_decay=decay, nesterov=True)

# Trainer object used for training
my_trainer = trainer.CompleteDocumentTrainer(train_iterator, myModel, cuda, optimizer)

# Evaluator
my_eval = trainer.EvaluatorFactory.get_evaluator("cross_entropy", cuda)
# Running epochs_class epochs
for epoch in range(0, epochs):
    logger.info("Epoch : %d", epoch)
    my_trainer.update_lr(epoch, schedule, gammas)
    my_trainer.train(epoch)
    my_eval.evaluate(my_trainer.model, val_iterator, epoch, prefix="val_",table=False)
my_eval.evaluate(my_trainer.model, test_iterator, epoch, prefix="test_",table=True)

torch.save(myModel.state_dict(), my_experiment.path + dataset_type + "_" + model_type + ".pb")
my_experiment.store_json()
wandb.finish()

#%%
