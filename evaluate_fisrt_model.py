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

experiment_names = "Experiment-7-doc"

output_dir = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\experiments"
no_cuda = False
data_dirs = [
    "/home/ubuntu/document_localization/Recursive-CNNs/datasets/augmentations",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/smart-doc-train",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/self_collected",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/kosmos"
]
dataset_type = "document"
validation_dirs = [
    "/home/ubuntu/document_localization/Recursive-CNNs/datasets/augmentations",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/smart-doc-train",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/self_collected",
    # "/home/ubuntu/document_localization/Recursive-CNNs/datasets/kosmos"
]
loader = "ram"

model_type = "resnet"

pretrain = False

lr = 0.005
batch_size = 500
seed = 42

decay = 0.00001

epochs = 75
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
    "decay": decay,
    "epochs": epochs,
}
# wandb.login(key=[your_api_key])
wandb.init(project="document-detection",
           entity="kosmos-randd",
           config=arguments)

wandb.run.name = "evaluation"+ "-" + experiment_names
# Define an experiment.
my_experiment = ex.experiment(experiment_names, arguments, output_dir)

# Add logging support
logger = utils.utils.setup_logger(my_experiment.path)

#%%
dataset_val = dataprocessor.DatasetFactory.get_dataset(data_dirs, dataset_type, "train.csv")



val_dataset_loader = dataprocessor.LoaderFactory.get_loader(loader, dataset_val.myData,
                                                            transform=dataset_val.test_transform,
                                                            cuda=cuda)
kwargs = {'num_workers': 30, 'pin_memory': True} if cuda else {}

val_iterator = torch.utils.data.DataLoader(val_dataset_loader,
                                           batch_size=batch_size, shuffle=True, **kwargs)

# Get the required model
myModel = model.ModelFactory.get_model(model_type, dataset_type)

myModel.load_state_dict(torch.load(r"/home/ubuntu/document_localization/Recursive-CNNs/experiments3082024/Experiment-3-doc_0/Experiment-3-docdocument_resnet.pb", map_location='cpu'))


my_eval = trainer.EvaluatorFactory.get_evaluator("rmse", cuda)


# Final evaluation on test set
my_eval.evaluate(model, val_iterator, 0, "test_", True)

# torch.save(myModel.state_dict(), my_experiment.path + dataset_type + "_" + model_type + ".pb")
# my_experiment.store_json()
wandb.finish()
