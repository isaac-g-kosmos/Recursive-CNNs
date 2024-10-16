''' Pytorch Recursive CNN Trainer
 Authors : Khurram Javed
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function
from typing import Tuple, List, Dict

import logging

from torch.autograd import Variable

logger = logging.getLogger('iCARL')
import torch.nn.functional as F
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self):
        pass



class Trainer(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]

    def train(self, epoch,):
        self.model.train()
        logging_batch=epoch*len(self.train_iterator)
        lossAvg = None
        for img, target,_ in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda()
            self.optimizer.zero_grad()
            response = self.model(Variable(img))
            # print (response[0])
            # print (target[0])
            loss = F.mse_loss(response, Variable(target.float()))
            loss = torch.sqrt(loss)
            wandb.log({"batch": logging_batch, "batch_training_loss":loss.cpu().data.numpy()})
            logging_batch+=1
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg += loss
            # logger.debug("Cur loss %s", str(loss))
            loss.backward()
            self.optimizer.step()

        lossAvg /= len(self.train_iterator)
        lossAvg=(lossAvg).cpu().data.numpy()
        logger.info("Avg Loss %s", str(lossAvg))
        wandb.log({"epoch": epoch, "avg_train_loss": lossAvg})



class Trainer_with_class(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer,loss="mse", leeway=-1):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer
        self.leeway = leeway
        if loss=="mse":
            self.loss_funct=F.mse_loss
        elif loss=="l1":
            self.loss_funct=F.l1_loss

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]

    def train(self, epoch,):
        self.model.train()
        logging_batch=epoch*len(self.train_iterator)
        lossAvg = None
        classification_results=[]

        for img, target,_ in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda()
            self.optimizer.zero_grad()
            model_prediction = self.model(Variable(img))
            x_cords = model_prediction[:, [0, 2, 4, 6]]
            y_cords = model_prediction[:, [1, 3, 5, 7]]
            classification_result = self.evaluate_corners(x_cords, y_cords, target, _)
            classification_results.extend(classification_result)
            loss = self.loss_funct(model_prediction, Variable(target.float()))
            loss = torch.sqrt(loss)
            wandb.log({"batch": logging_batch, "batch_training_loss":loss.cpu().data.numpy()})
            logging_batch+=1
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg += loss
            # logger.debug("Cur loss %s", str(loss))
            loss.backward()
            self.optimizer.step()

        lossAvg /= len(self.train_iterator)
        lossAvg=(lossAvg).cpu().data.numpy()
        logger.info("Avg Loss %s", str(lossAvg))
        # wandb.log({"epoch": epoch, "avg_train_loss": lossAvg})
        df = pd.DataFrame(classification_results)
        # lossAvg /= len(iterator)
        total_corners = df["total_corners"]
        wandb.log({"epoch": epoch,
                   "avg_train_loss": lossAvg,
                   "train_accuracy": (np.sum(total_corners)/4)/len(total_corners),
                   "train_4_corners_accuracy": np.sum(total_corners == 4) / len(total_corners),
                   "train_3_corners_accuracy": np.sum(total_corners >= 3) / len(total_corners),

                   })
    def cordinate_within_intervals(self, cordinate, x_interval, y_interval) -> int:

        is_within_x = (x_interval[0] <= cordinate[0] <= x_interval[1])
        is_within_y = (y_interval[0] <= cordinate[1] <= y_interval[1])

        return int(is_within_x and is_within_y)

    def euclidean_distance_np(self,x1, y1, x2, y2):
        # Convert to NumPy arrays if they aren't already
        x1, y1, x2, y2 = np.array(x1), np.array(y1), np.array(x2), np.array(y2)
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_area_vect(self,y_lower_bound, y_upper_bound, x_lower_bound, x_upper_bound):
        width = x_upper_bound - x_lower_bound
        height = y_upper_bound - y_lower_bound

        width = np.maximum(0, width)
        height = np.maximum(0, height)

        # Vectorized area calculation
        return width * height

    def evaluate_corners(self, x_cords: np.ndarray, y_cords: np.ndarray, target: np.ndarray,paths:str,
                         leeway) -> Dict:

        target = target.cpu().data.numpy()
        x_cords = x_cords.cpu().data.numpy()
        y_cords = y_cords.cpu().data.numpy()

        x0, x1, x2, x3 = x_cords[:, 0], x_cords[:, 1], x_cords[:, 2], x_cords[:, 3]
        y0, y1, y2, y3 = y_cords[:, 0], y_cords[:, 1], y_cords[:, 2], y_cords[:, 3]

        target_x = target[:, [0, 2, 4, 6]]
        target_y = target[:, [1, 3, 5, 7]]

        doc_width = (self.euclidean_distance_np(x0, y0, x1, y1) + self.euclidean_distance_np(x3, y3, x2, y2)) / 2
        doc_height = (self.euclidean_distance_np(x0, y0, x3, y3) + self.euclidean_distance_np(x1, y1, x2, y2)) / 2


        # Top-left bounds
        top_left_y_lower_bound = np.maximum(0, (2 * y0 - (y3 + y0) / 2))
        top_left_y_upper_bound = (y3 + y0) / 2
        top_left_x_lower_bound = np.maximum(0, (2 * x0 - (x1 + x0) / 2))
        top_left_x_upper_bound = (x1 + x0) / 2

        # Top-right bounds
        top_right_y_lower_bound = np.maximum(0, (2 * y1 - (y1 + y2) / 2))
        top_right_y_upper_bound = (y1 + y2) / 2
        top_right_x_lower_bound = (x1 + x0) / 2
        top_right_x_upper_bound = np.minimum(1, (x1 + (x1 - x0) / 2))

        # Bottom-right bounds
        bottom_right_y_lower_bound = (y1 + y2) / 2
        bottom_right_y_upper_bound = np.minimum(1, (y2 + (y2 - y1) / 2))
        bottom_right_x_lower_bound = (x2 + x3) / 2
        bottom_right_x_upper_bound = np.minimum(1, (x2 + (x2 - x3) / 2))

        # Bottom-left bounds
        bottom_left_y_lower_bound = (y0 + y3) / 2
        bottom_left_y_upper_bound = np.minimum(1, (y3 + (y3 - y0) / 2))
        bottom_left_x_lower_bound = np.maximum(0, (2 * x3 - (x2 + x3) / 2))
        bottom_left_x_upper_bound = (x3 + x2) / 2

        partitions_dictionary = {
            "top_left": [top_left_y_lower_bound, top_left_y_upper_bound, top_left_x_lower_bound,
                         top_left_x_upper_bound, (x_cords[:, 0], y_cords[:, 0])],
            "top_right": [top_right_y_lower_bound, top_right_y_upper_bound, top_right_x_lower_bound,
                          top_right_x_upper_bound, (x_cords[:, 1], y_cords[:, 1])],
            "bottom_right": [bottom_right_y_lower_bound, bottom_right_y_upper_bound, bottom_right_x_lower_bound,
                             bottom_right_x_upper_bound, (x_cords[:, 2], y_cords[:, 2])],
            "bottom_left": [bottom_left_y_lower_bound, bottom_left_y_upper_bound, bottom_left_x_lower_bound,
                            bottom_left_x_upper_bound, (x_cords[:, 3], y_cords[:, 3])]
        }


        for key in partitions_dictionary.keys():
            current_bb = partitions_dictionary[key]
            predicted_cordinates = current_bb[4]

            new_y_lower_bound = current_bb[0]
            new_y_upper_bound = current_bb[1]
            new_x_lower_bound = current_bb[2]
            new_x_upper_bound = current_bb[3]

            BB_area_boolean_mask = self.calculate_area_vect(
                new_y_lower_bound,
                new_y_upper_bound,
                new_x_lower_bound,
                new_x_upper_bound) < .05

            new_y_lower_bound[BB_area_boolean_mask] = predicted_cordinates[1][BB_area_boolean_mask] - leeway * doc_height[
                BB_area_boolean_mask]
            new_y_upper_bound[BB_area_boolean_mask] = predicted_cordinates[1][BB_area_boolean_mask] + leeway * doc_height[
                BB_area_boolean_mask]
            new_x_lower_bound[BB_area_boolean_mask] = predicted_cordinates[0][BB_area_boolean_mask] - leeway * doc_width[BB_area_boolean_mask]
            new_x_upper_bound[BB_area_boolean_mask] = predicted_cordinates[0][BB_area_boolean_mask] + leeway * doc_width[BB_area_boolean_mask]

            partitions_dictionary[key] = [
                new_y_lower_bound,
                new_y_upper_bound,
                new_x_lower_bound,
                new_x_upper_bound,
                predicted_cordinates
            ]

        tl = (partitions_dictionary["top_left"][0] <= target_y[:, 0]) & (
                    target_y[:, 0] <= partitions_dictionary["top_left"][1]) & \
             (partitions_dictionary["top_left"][2] <= target_x[:, 0]) & (
                         target_x[:, 0] <= partitions_dictionary["top_left"][3])

        tr = (partitions_dictionary["top_right"][0] <= target_y[:, 1]) & (
                    target_y[:, 1] <= partitions_dictionary["top_right"][1]) & \
             (partitions_dictionary["top_right"][2] <= target_x[:, 1]) & (
                         target_x[:, 1] <= partitions_dictionary["top_right"][3])

        br = (partitions_dictionary["bottom_right"][0] <= target_y[:, 2]) & (
                    target_y[:, 2] <= partitions_dictionary["bottom_right"][1]) & \
             (partitions_dictionary["bottom_right"][2] <= target_x[:, 2]) & (
                         target_x[:, 2] <= partitions_dictionary["bottom_right"][3])

        bl = (partitions_dictionary["bottom_left"][0] <= target_y[:, 3]) & (
                    target_y[:, 3] <= partitions_dictionary["bottom_left"][1]) & \
             (partitions_dictionary["bottom_left"][2] <= target_x[:, 3]) & (
                         target_x[:, 3] <= partitions_dictionary["bottom_left"][3])
                   
        tl = tl.astype('int')
        tr = tr.astype('int')
        br = br.astype('int')
        bl = bl.astype('int')
       
        result_dicts=[]
        for idx in range(len(paths)):

            result_dict = {"path":paths[idx],
                "contains_tl": tl[idx],
                           "contains_tr": tr[idx],
                           "contains_br": br[idx],
                           "contains_bl": bl[idx],
                           "total_corners": tl[idx] + tr[idx] + br[idx] + bl[idx],

                "top_left": ((partitions_dictionary["top_left"][4][0][idx],partitions_dictionary["top_left"][4][1][idx]),
                             (target_x[idx][0],target_y[idx][0]),
                             partitions_dictionary["top_left"][0][idx],
                             partitions_dictionary["top_left"][1][idx],
                             partitions_dictionary["top_left"][2][idx],
                             partitions_dictionary["top_left"][3][idx]),
                "top_right": ((partitions_dictionary["top_right"][4][0][idx],partitions_dictionary["top_right"][4][1][idx]),
                              (target_x[idx][1],target_y[idx][1]),
                              partitions_dictionary["top_right"][0][idx],
                              partitions_dictionary["top_right"][1][idx],
                              partitions_dictionary["top_right"][2][idx],
                              partitions_dictionary["top_right"][3][idx]),
                "bottom_right": ((partitions_dictionary["bottom_right"][4][0][idx],partitions_dictionary["bottom_right"][4][1][idx]),
                                 (target_x[idx][2],target_y[idx][2]),
                                 partitions_dictionary["bottom_right"][0][idx],
                                 partitions_dictionary["bottom_right"][1][idx],
                                 partitions_dictionary["bottom_right"][2][idx],
                                 partitions_dictionary["bottom_right"][3][idx]),
                "bottom_left": ((partitions_dictionary["bottom_left"][4][0][idx],partitions_dictionary["bottom_left"][4][1][idx]),
                                (target_x[idx][3],target_y[idx][3]),
                                partitions_dictionary["bottom_left"][0][idx],
                                partitions_dictionary["bottom_left"][1][idx],
                                partitions_dictionary["bottom_left"][2][idx],
                                partitions_dictionary["bottom_left"][3][idx]),
            }
            result_dicts.append(result_dict)
        return result_dicts
                          

class CompleteDocumentTrainer(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]

    def train(self, epoch,):
        self.model.train()
        logging_batch=epoch*len(self.train_iterator)
        lossAvg = None
        loss_function = torch.nn.CrossEntropyLoss()
        for img, target ,type,dataset,path in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda()
            self.optimizer.zero_grad()
            response = self.model(Variable(img))

            loss =loss_function(response, Variable(target.float()))

            wandb.log({"batch": logging_batch, "batch_training_loss":loss.cpu().data.numpy()})
            logging_batch+=1
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg += loss

            loss.backward()
            self.optimizer.step()

        lossAvg /= len(self.train_iterator)
        lossAvg=(lossAvg).cpu().data.numpy()
        logger.info("Avg Loss %s", str(lossAvg))
        wandb.log({"epoch": epoch, "avg_train_loss": lossAvg})

class CIFARTrainer(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, targets in tqdm(self.train_iterator):
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(Variable(inputs), pretrain=True)
            loss = self.criterion(outputs, Variable(targets))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        logger.info("Accuracy : %s", str((correct * 100) / total))
        return correct / total
