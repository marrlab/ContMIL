from collections import defaultdict
import os
import pickle
from copy import deepcopy

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import random
from Dataloader import Dataloader
from model import AMiL, AMiLExpandable

# create prepare data ,train_loader, ..


class ContinualLearner:
    def __init__(self, task, method):
        print(task)
        self.task = task
        self.method = method.lower()

        self.epochs = 1

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.ngpu = torch.cuda.device_count()
        print("Found device: ", self.ngpu, "x ", self.device)
        self._init_model()
        # self.wa_weights = wa_weights
        print("Setup complete.")

    def _init_model(self):
        if self.task.task_id == 0:
            if self.method == "der":
                self.model = AMiLExpandable()
            else:
                self.model = AMiL()
            # set number of output classes
            self.model.classifier[-1] = nn.Linear(256,
                                                  len(self.task.class_list))
        else:
            # load the last model
            print("loading model-" + self.task.prev_modelname +
                  "-" + self.method + " for training")
            self.model = torch.load(
                os.path.join("models",
                             "model-" + self.task.prev_modelname + "-" + self.method + ".pt"),
                map_location="cpu")
            in_features = self.model.classifier[-1].in_features
            out_features = self.model.classifier[-1].out_features
            # save weights of the previous task
            weight = self.model.classifier[-1].weight.data
            # replace last layer of classifier with a new layer with cumm_cls outputs
            self.model.classifier[-1] = nn.Linear(
                in_features, len(self.task.cumm_cls), bias=False)
            # set known weights of previous classes and initialize the rest from scratch
            self.model.classifier[-1].weight.data[:out_features] = weight
            if self.task.experiment == 1:
                self.n_known = len(
                    [x for x in self.task.cumm_cls if x not in self.task.class_list])
            else:
                self.n_known = len(self.task.class_list)

        self.model = self.model.to(self.device)

    def train_task(self, data):
        print("training task")

        if self.method == "lb":
            self._train_LB(data)
        elif self.method == "ewc":
            self._train_ewc(data)
        elif self.method == "icarl":
            self._train_iCaRL(data, bic=False)
        elif self.method == "atticarl":
            self._train_attiCaRL(data)
        elif self.method == "csatticarl_ks":
            self._train_csattiCaRL(data, method='ks')
        elif self.method == "csatticarl_random":
            self._train_csattiCaRL(data, method='random')
        elif self.method == "lwf":
            self._train_lwf(data)
        elif self.method == "mas":
            self._train_mas(data)
        elif self.method == "wa":
            self._train_wa(data)
        elif self.method == "bic":
            self._train_iCaRL(data, bic=True)
        elif self.method == "rebalancing":
            self._train_iCaRL(data, bic=True)
        elif self.method == 'der':
            self._train_der(data)
        else:
            print("Method not found")
            exit()
        self.save_model()

    """
    ===============================================================================
    Method: DER
    -------------------------------------------------------------------------------
    ===============================================================================
    """

    def _init_der(self):
        self.exemplar_sets = {}
        self.cellsinfo = {}
        self.K = 5000
        self.dist_loss = nn.BCELoss()
        self.compute_means = True
        self.counter = 0
        # not sure ?
        self.s_max = 10
        if self.task.task_id > 0:
            filename = "tmp/" + str(self.task.experiment) + self.method + self.task.datasets[
                0] + ".tmp" if self.task.experiment == 4 \
                else "tmp/" + str(self.task.experiment) + self.method + ".tmp"
            with open(filename, "rb") as f:
                if self.method == "csatticarl":
                    self.exemplar_sets, self.cellsinfo = pickle.load(f)
                else:
                    self.exemplar_sets = pickle.load(f)

    def _train_der(self, data):
        print("Training task")
        self._init_der()

        # Step 1: Add a new feature extractor if this is not the first task
        if self.task.task_id > 0:
            self.add_new_feature_extractor(
                input_channels=512, output_channels=400, hidden_layers=[300, 200, 100])

        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)
        train_loader = torch.utils.data.DataLoader(dl, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)
        if self.task.task_id > 0:
            self.combine_dataset_with_exemplars(dl)
            dl = Dataloader(self.task, data, train=True)
            dlt = Dataloader(self.task, data, train=False)
            train_loader = torch.utils.data.DataLoader(dl, num_workers=1)
            test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)
            self.model.update_aux_classifier(len(self.task.class_list) + 1)

        # Step 3: Representation Learning Stage
        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.0005, momentum=0.9, nesterov=True)
        for ep in range(self.epochs):
            self.model.train()
            total_loss, corrects = 0, 0

            for batch_idx, (_, bag, label) in enumerate(train_loader):
                optimizer.zero_grad()
                label, bag = label.to(self.device), bag.to(
                    self.device).squeeze()

                # Forward pass
                prediction, current_task_features = self.model(bag)

                # Compute losses
                clsloss = nn.CrossEntropyLoss()(prediction, label)  # Classification loss (L_H^t)
                aux_loss = self._compute_aux_loss(
                    current_task_features, label) if current_task_features is not None else 0  # Auxiliary loss (L_Ha^t)
                sparsity_loss = self._compute_sparsity_loss()  # Sparsity loss (L_S)

                # Combine losses
                loss = clsloss + 0.1 * aux_loss + 0.1 * sparsity_loss
                total_loss += loss.item()

                # Backpropagation
                loss.backward()
                self._compensate_mask_gradients(batch_idx, len(train_loader))
                optimizer.step()

                if torch.argmax(prediction, dim=1).item() == label.item():
                    corrects += 1

            accuracy = corrects / len(train_loader)
            print(
                f"Epoch {ep + 1}/{self.epochs}, Loss: {total_loss:.3f}, Accuracy: {accuracy:.3f}")

        # Step 4: Prune and binarize the current feature extractor
        self.binarize_masks()
        # balance data
        bdl, _ = self.create_balanced_data_loader(dl)
        # Step 4: Classifier Learning Stage (Retrain Classifier with Balanced Data)
        self.retrain_classifier(bdl)
        self.update_memory(dl)
        # Step 6: Evaluate and finalize the model
        self._evaluate_model(test_loader)

    def _compute_aux_loss(self, current_task_features, labels):
        # Auxiliary classifier for the new feature extractor (F_t(x))
        aux_predictions = F.softmax(
            self.model.aux_classifier(current_task_features), dim=1)

        # Create auxiliary labels: new classes + 1 "other" category for old classes
        aux_labels = torch.where(labels >= len(
            self.task.class_list), len(self.task.class_list), labels)

        # Cross-entropy loss for auxiliary classifier
        aux_loss = nn.CrossEntropyLoss()(aux_predictions, aux_labels)
        return aux_loss

    def _compute_sparsity_loss(self):
        sparsity_loss = 0
        # Focus only on the current task's feature extractor (last one added)
        if len(self.model.additional_feature_extractors) > 0:
            current_extractor = self.model.additional_feature_extractors[-1]
            for name, param in current_extractor.named_parameters():
                if "mask" in name:
                    sparsity_loss += torch.sum(torch.abs(param))
        return sparsity_loss

    def update_memory(self, dl):
        m = self.K // (len(self.exemplar_sets.keys()) +
                       (len(self.task.class_list)))
        self.reduce_exemplar_sets(m)
        print("Retraining the classifier with balanced data...")
        for cls in self.task.class_list:
            print("Constructing exemplar set for " +
                  str(cls), end="... ", flush=True)
            imgs = dl.get_images(cls)
            self.exemplar_sets[str(
                cls)] = self.construct_exemplar_set(imgs, m)
            print("[done]")

        for eskey in self.exemplar_sets:
            print("Exemplar set: ", eskey, len(self.exemplar_sets[eskey]))

        filename = "tmp/" + str(self.task.experiment) + self.method + self.task.datasets[
            0] + ".tmp" if self.task.experiment == 4 \
            else "tmp/" + str(self.task.experiment) + self.method + ".tmp"
        os.makedirs("tmp", exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.exemplar_sets, f)

    def create_balanced_data_loader(self, dl):
        """
        Creates a balanced data loader using the existing Dataloader class.
        Args:
            task: Task instance with class and patient info.
            data: Dataset containing features and labels.
        Returns:
            list: Balanced dataset as a list of (features, label) tuples.
        """

        # Step 1: Group patient indices by class
        class_indices = defaultdict(list)
        for idx in range(len(dl)):
            _, _, label = dl[idx]
            class_indices[label].append(idx)

        # Step 2: Determine the minimum number of samples per class
        min_samples = min(len(indices) for indices in class_indices.values())

        # Step 3: Sample `min_samples` examples from each class
        balanced_data = []
        for indices in class_indices.values():
            sampled_indices = random.sample(indices, min_samples)
            balanced_data.extend(sampled_indices)

        # Step 4: Create a balanced dataset as a dictionary
        balanced_dataset = {}
        for idx in balanced_data:
            pid = dl.patientlist[idx]  # Retrieve pid
            feats = dl[idx][1]  # Retrieve features
            # Use pid as key and features as value
            balanced_dataset[pid] = feats
        temp_task = deepcopy(self.task)  # Make a copy of the task
        temp_task.patients = {"trainset": {},
                              "testset": {}}  # Reset patient sets
        temp_task.patients["trainset"] = {
            pid: dl.patients[pid] for pid in balanced_dataset.keys()}
        bdl = Dataloader(temp_task, balanced_dataset, train=True)
        btrain_loader = torch.utils.data.DataLoader(bdl, num_workers=1)
        return btrain_loader, balanced_data

    def _compensate_mask_gradients(self, batch_idx, num_batches):
        """Compensates gradients for mask parameters in the current feature extractor."""
        s = self.s_max * (batch_idx / (num_batches - 1))
        if len(self.model.additional_feature_extractors) > 0:
            current_extractor = self.model.additional_feature_extractors[-1]
            for name, param in current_extractor.named_parameters():
                if "mask" in name:
                    grad = param.grad
                    param.grad = grad * (s / (1 + s))

    def add_new_feature_extractor(self, input_channels, output_channels, hidden_layers):
        print("Adding new feature extractor...")
        self.model.add_feature_extractor(
            input_channels, output_channels, hidden_layers)

    def retrain_classifier(self, balanced_loader):
        """
        Retrain the classifier head with a class-balanced dataset.
        """
        print("Retraining the classifier...")

        # # Step 1: Reinitialize the classifier head
        # self.model.update_classifier()
        # Step 2: Prepare optimizer
        optimizer = optim.SGD(
            self.model.classifier.parameters(), lr=0.001, momentum=0.9)

        # Step 3: Retrain the classifier
        self.model.eval()  # Freeze feature extractors
        self.model.classifier.train()  # Train only the classifier

        for ep in range(1):  # Fewer epochs for classifier retraining
            total_loss, corrects = 0, 0

            for _, bag, label in balanced_loader:
                optimizer.zero_grad()
                label, bag = label.to(self.device), bag.to(
                    self.device).squeeze()

                # Forward pass through the frozen feature extractor
                with torch.no_grad():
                    features = self.model.get_features(bag)

                # Apply the classifier
                logits = self.model.classifier(features)
                clsloss = nn.CrossEntropyLoss()(logits, label)
                total_loss += clsloss.item()

                # Backpropagation
                clsloss.backward()
                optimizer.step()

                if torch.argmax(logits, dim=1).item() == label.item():
                    corrects += 1

            accuracy = corrects / len(balanced_loader)
            print(
                f"Classifier Retraining Epoch {ep + 1}: Loss: {total_loss:.3f}, Accuracy: {accuracy:.3f}")

    def binarize_masks(self):
        # Focus only on the current task's feature extractor (last one added)
        if len(self.model.additional_feature_extractors) > 0:
            current_extractor = self.model.additional_feature_extractors[-1]
            for name, param in current_extractor.named_parameters():
                if "mask" in name:
                    param.data = (param.data > 0.5).float()  # Binarize masks

    def _evaluate_model(self, test_loader):
        self.model.eval()
        corrects = 0

        with torch.no_grad():
            for _, bag, label in test_loader:
                label, bag = label.to(self.device), bag.to(
                    self.device).squeeze()
                prediction, _ = self.model(bag)
                if torch.argmax(prediction, dim=1).item() == label.item():
                    corrects += 1

        accuracy = corrects / len(test_loader)
        print(f"Test Accuracy: {accuracy:.3f}")

    """
    ===============================================================================
    Method: WA
    -------------------------------------------------------------------------------
    ===============================================================================
    """

    def _train_wa(self, data):
        # Initialize WA weights for the first task if not already initialized
        if self.task.task_id == 0 and ContinualLearner.wa_weights is None:
            self._init_WA()

        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)
        train_loader = torch.utils.data.DataLoader(dl, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)

        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.0005, momentum=0.9, nesterov=True)

        for ep in range(self.epochs):
            self.model.train()
            corrects = 0
            totloss = 0

            for _, bag, label in train_loader:
                optimizer.zero_grad()
                label = label.to(self.device)
                bag = bag.to(self.device).squeeze()
                prediction = self.model(bag)
                loss = nn.NLLLoss()(nn.LogSoftmax(1)(prediction), label)
                totloss += loss.data
                loss.backward()
                optimizer.step()
                if torch.argmax(prediction, dim=1).item() == label.item():
                    corrects += 1

            samples = len(train_loader)
            totloss /= samples
            accuracy = corrects / samples

            print(
                f"- epoch: {ep + 1}/{self.epochs}, loss: {totloss:.3f}, acc: {accuracy:.3f}", end=' ')

            # Evaluation
            self.model.eval()
            corrects = 0
            for _, bag, label in test_loader:
                label = label.to(self.device)
                bag = bag.squeeze().to(self.device)
                prediction = self.model(bag)
                if torch.argmax(prediction, dim=1).item() == label.item():
                    corrects += 1

            accuracy = corrects / len(test_loader)
            print(f'test_acc: {accuracy:.3f}')

        # Update WA weights
        self._align_weights()

        # Update WA weights for the next task
        ContinualLearner.wa_weights = {n: deepcopy(
            p) for n, p in self.model.named_parameters() if p.requires_grad}

    def _init_WA(self):
        # Store initial weights for aligning
        ContinualLearner.wa_weights = {n: deepcopy(
            p) for n, p in self.model.named_parameters() if p.requires_grad}

    def _align_weights(self):
        # Align new weights with the old weights
        if ContinualLearner.wa_weights is None:
            print("No WA weights found for alignment.")
            return

        # Align new weights with the old weights
        norms_old = []
        norms_new = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if "classifier" in n:  # Only align classifier weights
                    if self.task.task_id > 0 and "weight" in n:
                        norms_old.append(torch.norm(
                            ContinualLearner.wa_weights[n].data))
                        norms_new.append(torch.norm(p.data))

        if norms_old and norms_new:
            mean_norm_old = torch.mean(torch.tensor(norms_old))
            mean_norm_new = torch.mean(torch.tensor(norms_new))
            gamma = mean_norm_old / (mean_norm_new + 1e-8)

            # Apply the scaling factor to new weights
            for n, p in self.model.named_parameters():
                if p.requires_grad and "classifier" in n and "weight" in n:
                    p.data *= gamma
    """
    ===============================================================================
    Method: LB
    -------------------------------------------------------------------------------
    Description:
    - Provide a brief overview of the method's functionality and purpose.
    - Highlight any key parameters or expected behavior.
    ===============================================================================
    """

    def _train_LB(self, data):
        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)
        train_loader = torch.utils.data.DataLoader(dl, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)

        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.0005, momentum=0.9, nesterov=True)

        for ep in range(self.epochs):
            self.model.train()
            corrects = 0
            totloss = 0

            for _, bag, label in train_loader:
                optimizer.zero_grad()

                # send to gpu
                label = label.to(self.device)
                bag = bag.to(self.device).squeeze()
                prediction = self.model(bag)

                loss = nn.NLLLoss()(nn.LogSoftmax(1)(prediction), label)
                totloss += loss.data

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            samples = len(train_loader)
            totloss /= samples

            accuracy = corrects / samples

            print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}'.format(
                ep + 1, self.epochs, totloss.cpu().detach().numpy(),
                accuracy), end=' ')

            self.model.eval()

            corrects = 0

            for _, bag, label in test_loader:
                label = label.to(self.device)
                bag = bag.squeeze().to(self.device)
                prediction = self.model(bag)
                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            accuracy = corrects / len(test_loader)
            print('test_acc: {:.3f}'.format(accuracy))
    """
    ===============================================================================
    Method: EWC
    -------------------------------------------------------------------------------
    Description:
    - Provide a brief overview of the method's functionality and purpose.
    - Highlight any key parameters or expected behavior.
    ===============================================================================
    """

    def _init_EWC(self, data):

        self.ewc_importance = 3000
        self.epochs = 10
        if self.task.task_id == 0:
            return

        dl = Dataloader(self.task, data, train=True)

        train_loader = torch.utils.data.DataLoader(dl, batch_size=1)

        self.ewc_params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.ewc_means = {}
        self.ewc_precision_matrices = self._diag_fisher(train_loader)

        for n, p in deepcopy(self.ewc_params).items():
            self.ewc_means[n] = Variable(p.data)

        del train_loader, dl

    def _diag_fisher(self, dataset):
        precision_matrices = {}
        for n, p in deepcopy(self.ewc_params).items():
            p.data.zero_()
            precision_matrices[n] = Variable(p.data)

        self.model.train()
        for _, img, lbl in dataset:
            self.model.zero_grad()
            input = Variable(img)
            output = self.model(input.float().squeeze().to(self.device))
            label = output.max(1)[1]
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def _ewc_penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            _loss = self.ewc_precision_matrices[n] * \
                (p - self.ewc_means[n]) ** 2
            loss += _loss.sum()
        return loss

    def _train_ewc(self, data):
        self._init_EWC(data)
        lam = 0.9
        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)
        train_loader = torch.utils.data.DataLoader(dl, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)

        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.0005, momentum=0.9, nesterov=True)

        for ep in range(self.epochs):
            self.model.train()
            corrects = 0
            train_loss = torch.tensor(0.).to(self.device)
            acc_cls_loss = torch.tensor(0.).to(self.device)
            acc_pen_loss = torch.tensor(0.).to(self.device)

            for _, bag, label in train_loader:
                optimizer.zero_grad()

                # send to gpu
                label = label.to(self.device)
                bag = bag.to(self.device).squeeze()
                prediction = self.model(bag)

                clsloss = nn.NLLLoss()(nn.LogSoftmax(1)(prediction), label)

                if self.task.task_id > 0:
                    ploss = self.ewc_importance * self._ewc_penalty()
                    acc_pen_loss += ploss.data
                    loss = (1 - lam) * clsloss + lam * ploss
                else:
                    loss = clsloss

                train_loss += loss.data
                acc_cls_loss += clsloss.data
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            samples = len(train_loader)
            train_loss /= samples
            acc_cls_loss /= samples
            acc_pen_loss /= samples

            accuracy = corrects / samples

            print("- epoch: {}/{}, loss:{:.3f}, cls_loss:{:.3f}, pen_loss:{:.3f}, acc: {:.3f}".format(
                ep + 1,
                self.epochs,
                train_loss.cpu().numpy(),
                acc_cls_loss.cpu().numpy(),
                acc_pen_loss.cpu().numpy(),
                accuracy), flush=True, end=" ")

            self.model.eval()

            corrects = 0

            for _, bag, label in test_loader:
                label = label.to(self.device)
                bag = bag.squeeze().to(self.device)
                prediction = self.model(bag)
                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            accuracy = corrects / len(test_loader)
            print('test_acc: {:.3f}'.format(accuracy))
    """
    ===============================================================================
    Method: iCARL 
    -------------------------------------------------------------------------------
    Description:
    - Provide a brief overview of the method's functionality and purpose.
    - Highlight any key parameters or expected behavior.
    ===============================================================================
    """

    def _init_icarl(self):
        self.exemplar_sets = {}
        self.cellsinfo = {}
        self.K = 5000
        self.dist_loss = nn.BCELoss()
        self.compute_means = True
        self.counter = 0

        if self.task.task_id > 0:
            filename = "tmp/" + str(self.task.experiment) + self.method + self.task.datasets[
                0] + ".tmp" if self.task.experiment == 4 \
                else "tmp/" + str(self.task.experiment) + self.method + ".tmp"
            with open(filename, "rb") as f:
                if self.method == "csatticarl":
                    self.exemplar_sets, self.cellsinfo = pickle.load(f)
                else:
                    self.exemplar_sets = pickle.load(f)

    def _train_iCaRL(self, data, bic=True):
        self._init_icarl()

        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)

        self.update_representation(data, dl, dlt, bic)
        m = self.K // (len(self.exemplar_sets.keys()) +
                       (len(self.task.class_list)))
        self.reduce_exemplar_sets(m)

        if self.task.experiment == 2:
            for cls in self.task.class_list:
                print("Constructing exemplar set for " +
                      str(self.task.task_id)+"-"+str(cls), end="... ", flush=True)
                imgs = dl.get_images(cls)
                self.exemplar_sets[str(
                    self.task.task_id)+"-"+str(cls)] = self.construct_exemplar_set(imgs, m)
                print("[done]")
        else:
            for cls in self.task.class_list:
                print("Constructing exemplar set for " +
                      str(cls), end="... ", flush=True)
                imgs = dl.get_images(cls)
                self.exemplar_sets[str(
                    cls)] = self.construct_exemplar_set(imgs, m)
                print("[done]")

        for eskey in self.exemplar_sets:
            print("Exemplar set: ", eskey, len(self.exemplar_sets[eskey]))

        filename = "tmp/" + str(self.task.experiment) + self.method + self.task.datasets[
            0] + ".tmp" if self.task.experiment == 4 \
            else "tmp/" + str(self.task.experiment) + self.method + ".tmp"
        os.makedirs("tmp", exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.exemplar_sets, f)

    def reduce_exemplar_sets(self, m):
        for eskey in self.exemplar_sets:
            budget = m
            oldlist = self.exemplar_sets[eskey]
            self.exemplar_sets[eskey] = []
            i = 0
            while budget > 0:
                self.exemplar_sets[eskey].append(oldlist[i])
                budget -= len(oldlist[i].squeeze())
                i += 1

    # def combine_dataset_with_exemplars(self, dataset):
    #     for eskey in self.exemplar_sets:
    #         if self.task.experiment == 2:
    #             eslbl = eskey.split("-")[1]
    #         else:
    #             eslbl = eskey

    #         dataset.append(self.exemplar_sets[eskey], eslbl)
    #     return dataset

    def construct_exemplar_set(self, images, m):
        # Compute and cache features for each example
        # load features of cls
        features = []
        for i, patient in enumerate(images):
            x = Variable(torch.tensor(patient).float()).to(self.device)
            feature = self.model.get_features(x.squeeze()).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)  # Normalize
            features.append(feature[0])
        # calculate class mean
        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        # herding selection
        exemplar_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        budget = m
        for k in range(m):
            if budget < 0:
                break
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
            budget = budget - len(images[i].squeeze())
            exemplar_set.append(np.expand_dims(images[i], 0))
            exemplar_features.append(features[i])
            """
            print "Selected example", i
            print "|exemplar_mean - class_mean|:",
            print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
            #features = np.delete(features, i, axis=0)
            """
        return exemplar_set

    # caching the network's pre-update outputs for distillation loss.
    def update_representation(self, data, dl, dlt, bic):
        self.compute_means = True
        # update dl with old data
        dl = self.combine_dataset_with_exemplars(dl)
        if bic:
            self.n_known = len(self.task.class_list)
            train_set, val_set = self.create_balanced_val_set(dl)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=1)
            test_loader = torch.utils.data.DataLoader(dlt, batch_size=1)
        else:
            train_loader = torch.utils.data.DataLoader(dl, batch_size=1)
            test_loader = torch.utils.data.DataLoader(dlt, batch_size=1)

        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.0005, momentum=0.9, nesterov=True)

        # Store network outputs with pre-update parameters
        q = torch.zeros(len(dl), len(self.task.cumm_cls)).to(self.device)
        for indices, images, labels in tqdm(train_loader, desc="Caching network outputs"):
            images = Variable(images.float().squeeze()).to(self.device)
            indices = indices.to(self.device)
            g = torch.sigmoid(self.model(images))
            q[indices] = g.data
        q = Variable(q).to(self.device)

        for ep in range(self.epochs):
            self.model.train()
            corrects = 0
            train_loss = torch.tensor(0.).to(self.device)
            # classification loss
            acc_cls_loss = torch.tensor(0.).to(self.device)
            # distillation loss
            acc_dist_loss = torch.tensor(0.).to(self.device)

            for indices, bag, label in train_loader:
                optimizer.zero_grad()

                # send to gpu
                label = label.to(self.device)
                bag = bag.to(self.device).squeeze()
                prediction = self.model(bag)
                # old+new
                clsloss = nn.NLLLoss()(nn.LogSoftmax(1)(prediction), label)

                # Distilation loss
                if self.task.task_id > 0:
                    # prediction on new tasks
                    g = torch.sigmoid(prediction)
                    # old task predictions
                    q_i = q[indices]
                    # loss for only old classes (y)
                    distloss = sum(self.dist_loss(g[:, y], q_i[:, y])
                                   for y in range(self.n_known))
                    # dist_loss = dist_loss / self.n_known
                    loss = clsloss + 5 * distloss
                    acc_dist_loss += distloss.data
                else:
                    loss = clsloss

                train_loss += loss.data
                acc_cls_loss += clsloss.data
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            samples = len(train_loader)
            train_loss /= samples
            acc_cls_loss /= samples
            acc_dist_loss /= samples

            accuracy = corrects / samples

            print("- epoch: {}/{}, loss:{:.3f}, cls_loss:{:.3f}, dist_loss:{:.3f}, acc: {:.3f}".format(
                ep + 1,
                self.epochs,
                train_loss.cpu().numpy(),
                acc_cls_loss.cpu().numpy(),
                acc_dist_loss.cpu().numpy(),
                accuracy), flush=True, end=" ")

        if bic:
            # Train bias correction layer
            self._train_bias_correction_layer(val_loader)

        self.model.eval()
        corrects = 0
        # TODO: ADD NME CLASSIFIER
        for _, bag, label in test_loader:
            label = label.to(self.device)
            bag = bag.squeeze().to(self.device)
            prediction = self.model(bag)
            if (torch.argmax(prediction, dim=1).item() == label.item()):
                corrects += 1

        accuracy = corrects / len(test_loader)
        print('test_acc: {:.3f}'.format(accuracy))

    def _train_bias_correction_layer(self, val_loader):
        """
        Trains the bias correction layer as described in the BiC paper.
        - Freezes the convolutional and fully connected layers.
        - Optimizes the bias parameters for new classes using a small validation set.
        """
        print("Training Bias Correction Layer...")

        # Step 1: Initialize bias correction parameters (if not already done)
        if not hasattr(self, 'bias_alpha') or not hasattr(self, 'bias_beta'):
            self.bias_alpha = nn.Parameter(torch.ones(
                1, requires_grad=True).to(self.device))  # Initialize α
            self.bias_beta = nn.Parameter(torch.zeros(
                1, requires_grad=True).to(self.device))  # Initialize β

        # Step 2: Freeze convolutional and fully connected layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Step 3: Define optimizer for bias parameters
        optimizer = optim.SGD(
            [self.bias_alpha, self.bias_beta], lr=0.001, momentum=0.9)

        # Step 4: Training loop for bias correction layer
        self.bias_alpha.requires_grad = True
        self.bias_beta.requires_grad = True

        for epoch in range(1):  # Single epoch as mentioned in the paper
            total_loss, corrects = 0, 0

            for _, bag, label in val_loader:
                bag, label = bag.to(self.device), label.to(self.device)
                bag = bag.squeeze()  # Remove unnecessary dimensions if present

                # Step 5: Compute logits for bias correction
                with torch.no_grad():
                    logits = self.model(bag)

                # Keep old logits as is and apply bias correction to new logits
                # Old classes (1, ..., n)
                old_logits = logits[:, :self.n_known]
                # New classes (n+1, ..., n+m)
                new_logits = logits[:, self.n_known:]
                corrected_new_logits = self.bias_alpha * new_logits + self.bias_beta

                # Combine old and corrected new logits
                corrected_logits = torch.cat(
                    [old_logits, corrected_new_logits], dim=1)

                # Step 6: Compute loss and update bias parameters
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(corrected_logits, label)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Accuracy calculation
                corrects += (corrected_logits.argmax(dim=1)
                             == label).sum().item()

            print(
                f"Bias Correction Epoch {epoch + 1}/1 - Loss: {total_loss / len(val_loader):.4f} - "
                f"Acc: {corrects / len(val_loader.dataset):.4f}"
            )

        print("Bias Correction Training Complete.")

        # Unfreeze all model parameters (if needed for future steps)
        for param in self.model.parameters():
            param.requires_grad = True

    def combine_dataset_with_exemplars(self, dataset):
        for eskey in self.exemplar_sets:
            if self.task.experiment == 2:
                eslbl = eskey.split("-")[1]
            else:
                eslbl = eskey
            dataset.append(self.exemplar_sets[eskey], eslbl)
        return dataset

    def create_balanced_val_set(self, dl):
        """
        Creates a balanced validation set for bias correction.
        For the first task, it considers only new classes.

        Args:
            dl: Dataloader instance that provides the data.

        Returns:
            train_set: Subset of training data.
            val_set: Subset of validation data.
        """
        from torch.utils.data import Subset

        # Step 1: Group patient indices by class
        class_indices = defaultdict(list)
        for idx in range(len(dl)):
            _, _, label = dl[idx]
            class_indices[label].append(idx)

        # Step 2: Initialize sets for trainold, valold, trainnew, and valnew
        trainold = []
        valold = []
        trainnew = []
        valnew = []

        if self.task.task_id == 0:
            # For the first task, consider only new classes
            for new_class in self.task.class_list:
                indices = class_indices[new_class]
                num_samples = len(indices)
                # Split into training and validation sets
                split_idx = num_samples // 2  # 50-50 split
                trainnew.extend(indices[:split_idx])
                valnew.extend(indices[split_idx:])
        else:
            # For subsequent tasks, split old and new classes separately
            for old_class in self.task.cumm_cls[:self.n_known]:
                indices = class_indices[old_class]
                num_samples = len(indices)
                # Split old class samples into train and validation sets
                split_idx = num_samples // 2  # 50-50 split
                trainold.extend(indices[:split_idx])
                valold.extend(indices[split_idx:])

            for new_class in self.task.cumm_cls[self.n_known:]:
                indices = class_indices[new_class]
                num_samples = len(indices)
                # Split new class samples into train and validation sets
                split_idx = num_samples // 2  # 50-50 split
                trainnew.extend(indices[:split_idx])
                valnew.extend(indices[split_idx:])

        # Step 3: Balance validation sets (valold and valnew)
        if self.task.task_id > 0:
            min_val_size = min(len(valold), len(valnew))
            valold = valold[:min_val_size]
            valnew = valnew[:min_val_size]

        # Step 4: Combine the training and validation sets
        train_indices = trainold + trainnew
        val_indices = valold + valnew

        # Step 5: Create subsets using the indices
        train_set = Subset(dl, train_indices)
        val_set = Subset(dl, val_indices)

        return train_set, val_set

    def classify(self, x):
        """Classify images by neares-means-of-exemplars
        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        batch_size = 1

        if self.task.experiment == 2:
            exemplar_sets = {}
            for eckey in self.exemplar_sets:
                exemplar_sets[eckey.split("-")[1]] = []
            for eckey in self.exemplar_sets:
                exemplar_sets[eckey.split(
                    "-")[1]].extend(self.exemplar_sets[eckey])
        else:
            exemplar_sets = self.exemplar_sets
        # computing exemplar means per class
        # TODO: move out mean calculation to end of each task?
        if self.compute_means:
            print("Computing mean of exemplars...")
            exemplar_means = []
            for eckey in exemplar_sets:
                features = []
                # Extract feature for each exemplar in P_y
                imgset = exemplar_sets[eckey]
                for ex in imgset:
                    ex = Variable(torch.tensor(
                        ex).float().squeeze()).to(self.device)
                    if len(ex.shape) == 3:
                        ex = ex.unsqueeze(0)
                    # get features
                    feature = self.model.get_features(ex).data
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        # (batch_size, n_classes, feature_size)
        means = torch.stack([means] * batch_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # feature = self.feature_extractor(x)  # (batch_size, feature_size)
        feature = self.model.get_features(x)

        for i in range(feature.size(0)):  # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        # (batch_size, feature_size, n_classes)
        feature = feature.expand_as(means)

        # .squeeze()  # (batch_size, n_classes)
        dists = (feature - means).pow(2).sum(1)
        _, preds = dists.min(1)

        return preds
    """
    ===============================================================================
    Method: att icarl 
    -------------------------------------------------------------------------------
    Description:
    - Provide a brief overview of the method's functionality and purpose.
    - Highlight any key parameters or expected behavior.
    ===============================================================================
    """

    def _train_attiCaRL(self, data):
        self._init_icarl()

        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)

        self.update_representation(dl, dlt)
        m = self.K // (len(self.exemplar_sets.keys()) +
                       (len(self.task.class_list)))
        self.reduce_exemplar_sets(m)

        if self.task.experiment == 2:
            for cls in self.task.class_list:
                print("Constructing exemplar set for " +
                      str(self.task.task_id)+"-"+str(cls), end="... ", flush=True)
                imgs = dl.get_images(cls)
                self.exemplar_sets[str(
                    self.task.task_id)+"-"+str(cls)] = self.construct_att_exemplerset(imgs, m)
                print("[done]")
        else:
            for cls in self.task.class_list:
                print("Constructing exemplar set for " +
                      str(cls), end="... ", flush=True)
                imgs = dl.get_images(cls)
                self.exemplar_sets[str(
                    cls)] = self.construct_att_exemplerset(imgs, m)
                print("[done]")

        for eskey in self.exemplar_sets:
            print("Exemplar set: ", eskey, len(self.exemplar_sets[eskey]))

        filename = "tmp/" + str(self.task.experiment) + self.method + self.task.datasets[
            0] + ".tmp" if self.task.experiment == 4 \
            else "tmp/" + str(self.task.experiment) + self.method + ".tmp"
        os.makedirs("tmp", exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.exemplar_sets, f)

    def construct_att_exemplerset(self, images, m):
        # Compute and cache features for each example
        features = []
        for i, patient in enumerate(images):
            x = Variable(torch.tensor(patient).float()).to(self.device)
            feature = self.model.get_features(x.squeeze()).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)  # Normalize
            features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        exemplar_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        budget = m
        for k in range(m):
            if budget < 0:
                break
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            # dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
            imgs, ind = self.get_high_att_cells(images[i], percent=0.1)
            budget = budget - len(imgs.squeeze())
            exemplar_set.append(np.expand_dims(imgs, 0))
            exemplar_features.append(features[i])

        # with open("tmp/att-cellselection_dump_" + str(self.counter) + ".pkl","wb") as f:
        #     pickle.dump([features, images, exemplar_set, exemplar_features],f)

        # self.counter += 1

        return exemplar_set

    def get_high_att_cells(self, images, percent):
        x = Variable(torch.tensor(images).float().squeeze()).to(self.device)
        _, att, _, _ = self.model.get_full_prediction(x)
        att = att.cpu().detach().numpy().squeeze()
        topn = int(np.ceil(len(att) * percent))
        ind = np.argpartition(att, -topn)[-topn:]
        ret = images[ind, :, :, :]

        return ret, ind
    """
    ===============================================================================
    Method: comil 
    -------------------------------------------------------------------------------
    Description:
    - Provide a brief overview of the method's functionality and purpose.
    - Highlight any key parameters or expected behavior.
    ===============================================================================
    """

    def _train_csattiCaRL(self, data, method='ks'):
        self._init_icarl()

        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)

        self.update_representation(dl, dlt)
        # nr of instances that get collected per class
        m = self.K // (len(self.exemplar_sets.keys()) +
                       (len(self.task.class_list)))
        self.reduce_exemplar_sets_ks(m)

        if self.task.experiment == 2:
            for cls in self.task.class_list:
                print("Constructing exemplar set for " +
                      str(cls), end="... ", flush=True)
                imgs = dl.get_images(cls)
                self.exemplar_sets[str(self.task.task_id)+"-"+str(cls)], self.cellsinfo[str(
                    self.task.task_id)+"-"+str(cls)] = self.construct_ks_exemplerset(imgs, m, method=method)
                print("[done]")
        else:
            for cls in self.task.class_list:
                print("Constructing exemplar set for " +
                      str(cls), end="... ", flush=True)
                imgs = dl.get_images(cls)
                self.exemplar_sets[str(cls)], self.cellsinfo[str(
                    cls)] = self.construct_exemplerset(imgs, m, method)
                print("[done]")

        for eskey in self.exemplar_sets:
            print("Exemplar set: ", eskey, len(self.exemplar_sets[eskey]))
        filename = "tmp/" + str(self.task.experiment) + self.method + self.task.datasets[
            0] + ".tmp" if self.task.experiment == 4 \
            else "tmp/" + str(self.task.experiment) + self.method + ".tmp"
        os.makedirs("tmp", exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump([self.exemplar_sets, self.cellsinfo], f)

    @staticmethod
    def knapsack(weights, values, capacity):
        n = len(weights)
        K = [[0.0 for j in range(capacity + 1)] for i in range(n + 1)]
        selected_items = []

        for i in range(1, n + 1):
            for j in range(1, capacity + 1):
                if weights[i - 1] <= j:
                    K[i][j] = max(values[i - 1] + K[i - 1]
                                  [int(j - weights[i - 1])], K[i - 1][j])
                else:
                    K[i][j] = K[i - 1][j]
        mmax_value = K[n][capacity]
        max_value = K[n][capacity]
        j = capacity

        for i in range(n, 0, -1):
            if max_value <= 0:
                break
            if max_value == K[i - 1][j]:
                continue
            else:
                selected_items.append(i - 1)
                max_value -= values[i - 1]
                j -= int(weights[i - 1])

        selected_items.reverse()

        return mmax_value, selected_items

    def construct_exemplerset(self, images, m, method):
        """
            Constructs an exemplar set using the specified selection method.

            Args:
                images: List of bags where each bag contains instances.
                m: Number of instances to select for the exemplar set.
                method: Selection method ("random" or "ks").

            Returns:
                exemplar_set: List of selected instances.
                cellsinfo: Cell-level information including bag ID, attention scores, distances, etc.
            """
        # compute the class mean
        features = []
        for i, patient in enumerate(images):
            x = Variable(torch.tensor(patient).float()).to(self.device)
            feature = self.model.get_features(x.squeeze()).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)  # Normalize
            features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize
        class_mean = torch.tensor(class_mean, device=self.device)

        cellsinfo = torch.tensor([], device=self.device)
        for i, patient in enumerate(images):
            x = Variable(torch.tensor(patient).float()).to(self.device)
            _, att, bagfeat, feats = self.model.get_full_prediction(
                x.squeeze())

            if len(feats.shape) < 2:
                feats = feats.unsqueeze(0)
                att = att.unsqueeze(0)
            assert len(feats.shape) == 2, feats.shape

            dist_classmean = torch.sum(F.mse_loss(feats, class_mean.repeat(len(feats), 1), reduction='none'),
                                       dim=1).unsqueeze(0)  # .cpu().detach().numpy()
            dist_bagfeat = torch.sum(F.mse_loss(feats, bagfeat.repeat(len(feats), 1), reduction='none'),
                                     dim=1).unsqueeze(0)  # .cpu().detach().numpy()
            bagid = torch.tensor(i, device=self.device).repeat(
                len(feats)).unsqueeze(0)
            cllid = torch.arange(
                0, len(feats), device=self.device).unsqueeze(0)
            cellsinfo = torch.concatenate(
                [cellsinfo, torch.concatenate([bagid, cllid, att, dist_bagfeat, dist_classmean, feats.T]).T])
        cellsinfo = cellsinfo.cpu().detach().numpy()
        # Step 3: Use the specified selection method
        if method == "random":
            exemplar_set, sampled_cellsinfo = self._select_random(
                images, cellsinfo, m)
        elif method == "ks":
            exemplar_set, sampled_cellsinfo = self._select_ks(
                images, cellsinfo, m)
        else:
            raise ValueError(
                f"Invalid method '{method}'. Use 'random' or 'ks'.")

        # exemplar_set, cellsinfo = self.select_exemplar_sets_ks(
        #     images, m, cellsinfo)

        # with open("tmp/ks-cellselection_dump_"+ str(self.counter) + ".pkl","wb") as f:
        #     pickle.dump([features, images, exemplar_set, cellsinfo],f)

        # self.counter += 1
        return exemplar_set, sampled_cellsinfo

    def _select_random(self, images, cellsinfo, m):
        """
        Randomly selects instances and their cell information.

        Args:
            cellsinfo: Cell information for all instances.
            m: Number of instances to select.

        Returns:
            exemplar_set: Randomly selected instances.
            sampled_cellsinfo: Corresponding cell information.
        """
        print("selecting random", end="... ", flush=True)
        # Step 1: Randomly select indices
        num_samples = min(m, len(cellsinfo))
        selected_items = random.sample(range(len(cellsinfo)), num_samples)

        # Step 2: Process the selected items
        exemplar_set = []
        simgs = np.array(cellsinfo[selected_items][:, :2], dtype=np.uint8)
        remaining_cells_info = []
        for i in np.unique(simgs[:, 0]):  # Group by bag ID
            assert len(images[i].shape) == 4, images[i].shape

            # Select images based on cell indices
            imgs = images[i][simgs[simgs[:, 0] == i, 1]]
            # Get corresponding cell info
            smpinfo = cellsinfo[selected_items][simgs[:, 0] == i]
            smpinfo[:, 0] = i  # Update bag ID
            # Update cell IDs within the bag
            smpinfo[:, 1] = np.array(range(0, imgs.shape[0]))
            exemplar_set.append(imgs)
            remaining_cells_info.append(smpinfo)

        return exemplar_set, remaining_cells_info

    def _select_ks(self, images, cellsinfo, m):
        """
        Selects instances using ks-based selection.

        Args:
            images: List of bags where each bag contains instances.
            cellsinfo: Cell information for all instances.
            m: Number of instances to select.

        Returns:
            exemplar_set: Instances selected by ks-based method.
            sampled_cellsinfo: Corresponding cell information.
        """

        exemplar_set, sampled_cellsinfo = self.select_exemplar_sets_ks(
            images, m, cellsinfo)
        return exemplar_set, sampled_cellsinfo

    @staticmethod
    def normalize(a):
        return (a - np.min(a)) / np.ptp(a)

    def select_exemplar_sets_ks(self, images, m, cellsinfo):

        print("solving KS", end="... ", flush=True)
        # cellsinfo = np.array(cellsinfo)
        cellscores = 0.3 * self.normalize(cellsinfo[:, 4]) + \
            0.2 * self.normalize(cellsinfo[:, 3]) + \
            0.5 * cellsinfo[:, 2]
        weights = np.ones(len(cellscores))
        _, selected_items = self.knapsack(weights, cellscores, m)
        # selected_items = np.array(range(m))
        exemplar_set = []
        simgs = np.array(cellsinfo[selected_items][:, :2], dtype=np.uint8)
        remaining_cells_info = []
        for i in np.unique(simgs[:, 0]):
            assert len(images[i].shape) == 4, images[i].shape

            imgs = images[i][simgs[simgs[:, 0] == i, 1]]
            smpinfo = cellsinfo[simgs[simgs[:, 0] == i, 1]]
            smpinfo[:, 0] = i
            smpinfo[:, 1] = np.array(range(0, imgs.shape[0]))
            exemplar_set.append(imgs)
            remaining_cells_info.append(smpinfo)

        return exemplar_set, remaining_cells_info

    def reduce_exemplar_sets_ks(self, m):
        for eskey in self.exemplar_sets:
            self.exemplar_sets[eskey], self.cellsinfo[eskey] = \
                self.reduce_exemplar_sets_ks_ins(
                    self.exemplar_sets[eskey], m, self.cellsinfo[eskey])

    def reduce_exemplar_sets_ks_ins(self, images, m, cellsinfo):
        exemplar_set = []
        cellinfo_set = []

        for imgs, cllinf in zip(images, cellsinfo):
            # dist class mean, dist bag mean, attention
            cellscores = 0.3 * self.normalize(cllinf[:, 4]) + \
                0.2 * self.normalize(cllinf[:, 3]) + \
                0.5 * cllinf[:, 2]
            weights = np.ones(len(cllinf))
            _, selected_items = self.knapsack(
                weights, cellscores, m//len(images))

            exemplar_set.append(imgs[selected_items])
            cellinfo_set.append(cllinf[selected_items])

        return exemplar_set, cellinfo_set
    """
    ===============================================================================
    Method: lwf
    -------------------------------------------------------------------------------
    Description:
    - Provide a brief overview of the method's functionality and purpose.
    - Highlight any key parameters or expected behavior.
    ===============================================================================
    """

    def _init_lwf(self):
        self.prev_model = deepcopy(self.model)
        self.prev_model.to(self.device)

    def _train_lwf(self, data):
        self._init_lwf()

        temperature = 2.0

        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)
        train_loader = torch.utils.data.DataLoader(dl, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)

        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.0005, momentum=0.9, nesterov=True)

        for ep in range(self.epochs):
            self.model.train()
            corrects = 0
            train_loss = torch.tensor(0.).to(self.device)
            acc_cls_loss = torch.tensor(0.).to(self.device)
            acc_pen_loss = torch.tensor(0.).to(self.device)

            for _, bag, label in train_loader:
                optimizer.zero_grad()

                # send to gpu
                label = label.to(self.device)
                bag = bag.to(self.device).squeeze()
                prediction = self.model(bag)

                clsloss = nn.NLLLoss()(nn.LogSoftmax(1)(prediction), label)

                if self.task.task_id > 0:
                    prev_outputs = self.prev_model(bag)
                    soft_targets = torch.softmax(
                        prev_outputs / temperature, dim=1)
                    distillation_loss = nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(prediction / temperature, dim=1),
                                                                            soft_targets.detach())

                    acc_pen_loss += distillation_loss.data
                    loss = clsloss + 0.5 * distillation_loss
                else:
                    loss = clsloss

                train_loss += loss.data
                acc_cls_loss += clsloss.data
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            samples = len(train_loader)
            train_loss /= samples
            acc_cls_loss /= samples
            acc_pen_loss /= samples

            accuracy = corrects / samples

            print("- epoch: {}/{}, loss:{:.3f}, cls_loss:{:.3f}, pen_loss:{:.3f}, acc: {:.3f}".format(
                ep + 1,
                self.epochs,
                train_loss.cpu().numpy(),
                acc_cls_loss.cpu().numpy(),
                acc_pen_loss.cpu().numpy(),
                accuracy), flush=True, end=" ")

            self.model.eval()

            corrects = 0

            for _, bag, label in test_loader:
                label = label.to(self.device)
                bag = bag.squeeze().to(self.device)
                prediction = self.model(bag)
                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            accuracy = corrects / len(test_loader)
            print('test_acc: {:.3f}'.format(accuracy))
    """
    ===============================================================================
    Method: mas
    -------------------------------------------------------------------------------
    Description:
    - Provide a brief overview of the method's functionality and purpose.
    - Highlight any key parameters or expected behavior.
    ===============================================================================
    """

    def _init_mas(self):
        self.mas_memory = [torch.zeros_like(
            p, requires_grad=False) for p in self.model.parameters()]
        self.prev_model = deepcopy(self.model)
        self.prev_model.to(self.device)
        self.lambd = 0.1
        self.alpha = 0.1

    def _train_mas(self, data):
        self._init_mas()

        dl = Dataloader(self.task, data, train=True)
        dlt = Dataloader(self.task, data, train=False)
        train_loader = torch.utils.data.DataLoader(dl, num_workers=1)
        test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)

        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.0005, momentum=0.9, nesterov=True)

        for ep in range(self.epochs):
            self.model.train()
            corrects = 0
            train_loss = torch.tensor(0.).to(self.device)
            acc_cls_loss = torch.tensor(0.).to(self.device)
            acc_pen_loss = torch.tensor(0.).to(self.device)

            for _, bag, label in train_loader:
                optimizer.zero_grad()
                self.mas_update_memory()

                # send to gpu
                label = label.to(self.device)
                bag = bag.to(self.device).squeeze()
                prediction = self.model(bag)

                clsloss = nn.NLLLoss()(nn.LogSoftmax(1)(prediction), label)

                if self.task.task_id > 0:
                    mas_loss = 0
                    for p, p_prev, mem in zip(self.model.parameters(), self.prev_model.parameters(), self.mas_memory):
                        if p.requires_grad:
                            mas_loss += torch.sum(mem * (p - p_prev) ** 2)
                    acc_pen_loss += mas_loss.data

                    loss = clsloss + self.lambd * mas_loss
                else:
                    loss = clsloss

                train_loss += loss.data
                acc_cls_loss += clsloss.data
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            samples = len(train_loader)
            train_loss /= samples
            acc_cls_loss /= samples
            acc_pen_loss /= samples

            accuracy = corrects / samples

            print("- epoch: {}/{}, loss:{:.3f}, cls_loss:{:.3f}, pen_loss:{:.3f}, acc: {:.3f}".format(
                ep + 1,
                self.epochs,
                train_loss.cpu().numpy(),
                acc_cls_loss.cpu().numpy(),
                acc_pen_loss.cpu().numpy(),
                accuracy), flush=True, end=" ")

            self.model.eval()

            corrects = 0

            for _, bag, label in test_loader:
                label = label.to(self.device)
                bag = bag.squeeze().to(self.device)
                prediction = self.model(bag)
                if (torch.argmax(prediction, dim=1).item() == label.item()):
                    corrects += 1

            accuracy = corrects / len(test_loader)
            print('test_acc: {:.3f}'.format(accuracy))

    def mas_update_memory(self):
        self.mas_memory = [torch.zeros_like(
            p, requires_grad=False) for p in self.model.parameters()]
        for p, p_prev in zip(self.model.parameters(), self.prev_model.parameters()):
            if p.requires_grad:
                delta = p.data - p_prev.data
                self.mas_memory += self.alpha * delta * delta

    def after_task(self, tm, data):
        accuracies = self._test_model(tm, data)
        if os.path.exists("results/" + str(self.task.experiment) + "-" + self.method + ".pkl"):
            with open("results/" + str(self.task.experiment) + "-" + self.method + ".pkl", "rb") as f:
                results = pickle.load(f)
        else:
            results = {}
        results[self.task.task_id] = accuracies
        os.makedirs("results", exist_ok=True)
        with open("results/" + str(self.task.experiment) + "-" + self.method + ".pkl", "wb") as f:
            pickle.dump(results, f)

    def _test_model(self, tm, data):
        ret = []
        for tid in range(self.task.task_id + 1):
            ret.append(self._test_model_on_task(tm.Tasks[tid], data))
        print(ret)
        return ret

    def _test_model_on_task(self, task, data):
        dlt = Dataloader(task, data, train=False)
        test_loader = torch.utils.data.DataLoader(dlt, num_workers=1)
        self.model.eval()
        gt = torch.tensor([]).to(self.device)
        preds = torch.tensor([]).to(self.device)
        for _, bag, label in test_loader:
            label = label.to(self.device).unsqueeze(0)
            bag = bag.squeeze().to(self.device)
            if self.method in ["icarl", "atticarl"]:
                prediction = self.classify(bag)
                preds = torch.cat([preds, prediction])
            elif self.method == "csatticarl":
                if task.task_id == self.task.task_id:
                    prediction = self.model(bag)
                    preds = torch.cat([preds, torch.argmax(prediction, dim=1)])
                else:
                    prediction = self.classify(bag)
                    preds = torch.cat([preds, prediction])
            elif self.method == 'der':
                prediction, _ = self.model(bag)
                preds = torch.cat([preds, torch.argmax(prediction, dim=1)])
            else:
                prediction = self.model(bag)
                preds = torch.cat([preds, torch.argmax(prediction, dim=1)])
            gt = torch.cat([gt, label])
        gt = gt.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(gt, preds)

        return accuracy

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        torch.save(self.model, os.path.join("models", "model-" +
                   self.task.modelname + "-" + self.method + ".pt"))
