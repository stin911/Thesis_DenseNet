from __future__ import print_function, division
import torch
import os
import torch
import pandas as pd
import warnings
import torch.optim as optim

warnings.filterwarnings("ignore")
from torchvision import transforms, utils
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import BMS_Loader as CDL
import monai as monai
import timeit
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from torchsummary import summary
from pathlib import Path
from sklearn.metrics import classification_report
import kornia.losses.focal as focal
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import Metric as metric
import Utility as ut
from collections import OrderedDict
import model_prova as mp
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


# Focal


# replace batch norm with instance norm

class BMSModel:
    """
    DenseNet model for binary classification 3D images

    setting:growth_rate=16, init_features=64
    loss: BCEWithLogitsLoss
    optim: Adam
    """

    def __init__(self, num_epochs, batch_size, lr, csv_train, csv_test, save_path, is_3d, num_workers=1):
        """

        :param num_epochs:
        :param batch_size:
        :param lr:
        :param csv_train: csv with images and label
        :param csv_test: csv with images and label
        :param num_workers:
        """
        # ----------- internal variable -------
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.csv_train = csv_train
        self.csv_test = csv_test
        # TODO boolean valuable cannot be passed through args parse
        self.is_3d = is_3d
        self.load = False

        # ------------------------list to save acc & loss
        self.t_loss = []
        self.t_acc = []
        self.v_acc = []
        self.v_loss = []

        self.best_auc = 0.55

        # ------------------------column label to save results
        self.column_name = ["Training True pos rate", 'Training False Positive rate',
                            'Training Specificify', "Validation True pos rate", 'Validation False Positive rate',
                            'Validation Specificify']
        self.result = pd.DataFrame(columns=self.column_name)

        # ---------------------------Data loading  and Dataloader creation--------------------------------------------
        self.data = CDL.Bms(csv_train, "D:/BoundingBoxBMS/",
                            transform=transforms.Compose([CDL.NormPad(), CDL.ToTensor()]), aug=True)
        self.val = CDL.Bms(csv_test, "D:/BoundingBoxBMS/",
                           transform=transforms.Compose([CDL.NormPad(), CDL.ToTensor()]))

        sampler = ut.weight_sampler(self.data)

        self.dataloader = DataLoader(self.data, batch_size=self.batch_size,
                                     num_workers=self.num_workers, sampler=sampler)
        self.val_loader = DataLoader(self.val, batch_size=1,
                                     shuffle=False, num_workers=self.num_workers)

        # -----------------------------------Set Device gpu or cpu-----------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------------------3D or 2D model----------------------------------------------------------------
        # Just for the Bms project .
        if self.is_3d:
            self.net = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1, out_channels=1, growth_rate=32,
                                                    init_features=128, block_config=(6, 12, 32, 32), dropout_prob=0.5,
                                                    ).to(self.device)
            print("3D model has been selected!...")
            if self.batch_size == 1:
                print(" Batch size is one ...Changing the BatchNorm to Instance Norm")
                ut.replace_batch_to(self.net, is_3d)
            self.net.to(self.device)
        else:
            print('2D model has been selected !...')
            self.net = monai.networks.nets.DenseNet(spatial_dims=2, in_channels=1, out_channels=1, growth_rate=32,
                                                    init_features=128,
                                                    dropout_prob=0.5).to(self.device)
            if self.batch_size == 1:
                print(" Batch size is one ...Changing the BatchNorm to Instance Norm")
                ut.replace_batch_to(self.net, is_3d)
            self.net.to(self.device)

        self.pos_weight = torch.Tensor([2]).to(self.device)  # Set pos-weight to balance the minority class
        # --------------------------Loss & optimizer ------------------------------------

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), self.lr)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)
        # self.optimizer = torch.optim.AdamW(self.net.parameters(), self.lr, weight_decay=0.5)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        # ---------------------------Load previous model and optimizer state-------------
        if self.load:

            ckp_path = "D:/Model_result_Bms/Pneumonia_Not/Pneumonia vs No Pneumonia/Slice Thickness Splitbest_74.pth"
            checkpoint = torch.load(ckp_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Number of sample in the training set :", self.data.__len__())
        print("Number of sample in the validation set :", self.val.__len__())

    def summary(self):
        """
        :return: generate a summary of the model
        """
        summary(self.net, (1, 80, 405, 507))

    def update_r(self, list_r):
        """
        :param list_r: list of the results of one epoch
        :return: updated csv of the result
        """

        p = pd.DataFrame([list_r], columns=self.column_name)

        self.result = self.result.append(p)

        self.result = self.result.round(5)

        self.result.to_csv(str(self.save_path) + "Result.csv")

    # --------------------------------Start the training ---------------------------
    def start_train(self, inference):
        """
        Start the training
        :return:
        """
        # ----------- Inference
        if inference:

            print("Validation")
            self.infer(self.val_loader, 'val_R')
            va_loss, v_true, v_pred, V_TPR, V_FPR, V_specificity, auc_val = self.train(False, self.val_loader)
            print("Train")
            va_loss, v_true, v_pred, V_TPR, V_FPR, V_specificity, auc_val = self.train(False, self.dataloader)
            self.infer(self.dataloader, 'train_R')
        # ----------- Training
        else:

            for epoch in range(74, self.num_epochs):
                print("Epoch: ", epoch)
                print('Train :')
                running_loss, y_true, y_pred, T_TPR, T_FPR, T_specificity, auc_val_t = self.train(True, self.dataloader)
                print('Validation :')
                va_loss, v_true, v_pred, V_TPR, V_FPR, V_specificity, auc_val_v = self.train(False, self.val_loader)

                res_list = [T_TPR, T_FPR, T_specificity, V_TPR, V_FPR, V_specificity]  # Update the result for the epoch
                self.update_r(res_list)

                if auc_val_v > self.best_auc and auc_val_t > 0.75:  # save weights if the condition hold
                    self.best_auc = auc_val_v
                    self.save_model_val(epoch)

                # --------------------Save the accuracy and loss for each epoch
                self.t_acc.append(accuracy_score(y_true, y_pred))
                self.t_loss.append(running_loss / self.data.__len__())
                self.v_acc.append(accuracy_score(v_true, v_pred))
                self.v_loss.append(va_loss / self.val.__len__())

                if epoch % 20 == 0:  # save model each 20 epochs
                    self.save_model(epoch)

                print("Accuracy on Training set is", accuracy_score(y_true, y_pred), 'Loss: ',
                      running_loss / self.data.__len__(),
                      "Accuracy on Validation set is", accuracy_score(v_true, v_pred), 'Loss: ',
                      va_loss / self.val.__len__())
                self.scheduler.step()  # LR scheduler step

            # TODO- save the list instead of print
            print('training loss:', self.t_loss)
            print('training acc:', self.t_acc)
            print('val loss:', self.v_loss)
            print('val acc:', self.v_acc)

    def train(self, train, data):
        """
        training function
        :return:
        """
        running_loss = 0
        y_true = []
        y_pred = []
        y_pred_prob = []
        if train:  # set the network to train
            self.net.train()
        else:  # set the network to eval
            self.net.eval()
        print('learning rate :', self.optimizer.param_groups[0]["lr"])
        for i_batch, sample_batched in enumerate(data):
            inputs, labels = sample_batched['image'].to(self.device), sample_batched['label'].to(self.device)
            if train:  # Training phase

                # self.optimizer.zero_grad()
                outputs = self.net(inputs)

                labels = labels.to(torch.float32)
                labels = labels.view(-1, 1)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()

                # self.optimizer.step()
                # gradient accumulation
                if (i_batch + 1) % 8 == 0:  # Gradient accumulation

                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:  # Validation phase
                with torch.no_grad():
                    labels = labels.to(torch.float32)
                    labels = labels.view(-1, 1)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()
            pred = np.round(torch.sigmoid(outputs.detach().cpu()).tolist())
            pred_prob = (torch.sigmoid(outputs.detach().cpu()).tolist())
            target = labels.tolist()
            y_true.extend(target[0])
            y_pred.extend(pred[0])
            y_pred_prob.extend(pred_prob[0])
        TPR, FPR, specificity = metric.roc_auc(y_pred, y_true)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=1)
        auc_val = auc(fpr, tpr)
        print("AUC Score : ", auc_val)

        return running_loss, y_true, y_pred, TPR, FPR, specificity, auc_val

    # Use it to make inference and save result to plot the roc_auc curve
    # TODO-- need to make order here
    def infer(self, dt, namecsv):

        """
        :return:
        """
        self.net.eval()
        va_loss = 0
        v_true = []
        v_pred = []
        name_l = []
        true_l = []
        pred_l = []

        for i_batch, sample_batched in enumerate(dt):
            inputs, labels, name = sample_batched['image'].to(self.device), sample_batched['label'].to(self.device), \
                                   sample_batched['name']
            with torch.no_grad():
                labels = labels.to(torch.float32)
                labels = labels.view(-1, 1)
                outputs = self.net(inputs)
                va_loss += self.criterion(outputs, labels).item()
                pred = (torch.sigmoid(outputs.detach().cpu()).tolist())  # Get the probability
                target = labels.tolist()
                name_l.extend(name)  # save image name
                v_true.extend(target[0])
                v_pred.extend(pred[0])

        fpr, tpr, thresholds = roc_curve(v_true, v_pred, pos_label=1)
        val = auc(fpr, tpr)
        print('AUC score :', val)
        metric.plot_roc_curve(fpr, tpr)
        average_precision = average_precision_score(v_true, v_pred)
        print("AP: Average precision : ", average_precision)
        precision, recall, thresholds = precision_recall_curve(v_true, v_pred)
        metric.plot_precis(precision, recall)
        dt = {'name': name_l, 'predict': v_pred, 'label': v_true}  # Save a report
        df = pd.DataFrame(data=dt)
        df.to_csv(self.save_path + str(
            namecsv) + '.csv')
        return va_loss, v_true, v_pred

    def get_feat(self):  # Save features of the last layer
        """
        :return:
        """
        self.net.eval()
        list_name = []
        list_f = []

        for i_batch, sample_batched in enumerate(self.dataloader):
            inputs, labels, name = sample_batched['image'].to(self.device), sample_batched['label'].to(self.device), \
                                   sample_batched['name']
            with torch.no_grad():
                outputs = self.net(inputs)
                list_f.append(((outputs.detach().cpu()).tolist())[0])

                list_name.append(name[0])
        df = {'name': list_name, 'feat': list_f}
        df = pd.DataFrame(data=df)
        print(df)
        df = pd.DataFrame(df['feat'].values.tolist()).add_prefix('feat').join(df['name'])
        df = round(df, 3)
        print(df)
        df.to_csv(self.save_path+"FeatureData.csv", index=False)

    def save_model(self, epoch):
        """
        Save the actual state of the network
        :param epoch:
        :return:
        """
        PATH = str(self.save_path) +'/'+ str(10 * epoch) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

    def save_model_val(self, epoch):
        """
        Save the actual state of the network
        :param epoch:
        :return:
        """
        PATH = str(self.save_path) + 'best_' + str(epoch) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),

        }, PATH)



