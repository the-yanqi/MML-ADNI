import torch
import torch.optim as optim
import torch.nn as nn
from time import time
from os import path
from copy import copy, deepcopy
import pandas as pd
import numpy as np
import torch.nn.init as init
import os


class CrossValidationSplit:
    """A class to create training and validation sets for a k-fold cross validation"""

    def __init__(self, dataset, cv=5, stratified=False):
        """
        :param cv: The number of folds to create
        :param stratified: Boolean to choose if we have the same repartition of diagnosis in each fold
        """
        self.cv = cv
        self.stratified = stratified
        subjects_df = dataset.subjects_df

        if stratified:
            n_diagnosis = len(dataset.diagnosis_code)
            cohorts = np.unique(dataset.subjects_df.cohort.values)

            folds_list = []
            for fold in range(cv):

                fold_list = []
                for cohort in cohorts:
                    for diagnosis in range(n_diagnosis):
                        diagnosis_key = list(dataset.diagnosis_code)[diagnosis]
                        diagnosis_df = subjects_df[(subjects_df.diagnosis == diagnosis_key) &
                                                   (subjects_df.cohort == cohort)]
                        fold_list.append(diagnosis_df.iloc[int(fold * len(diagnosis_df) / cv):int((fold + 1) * len(diagnosis_df) / cv):])
                fold_df = pd.concat(fold_list)
                folds_list.append(fold_df)

        else:
            folds_list = []
            for fold in range(cv):
                folds_list.append(subjects_df.iloc[int(fold * len(subjects_df) / cv):int((fold + 1) * len(subjects_df) / cv):])

        self.folds_list = folds_list
        self.iter = 0

    def __call__(self, dataset):
        """
        Calling creates a new training set and validation set depending on the number of times the function was called

        :param dataset: A dataset from data_loader.py
        :return: training set, validation set
        """
        if self.iter >= self.cv:
            raise ValueError('The function was already called %i times' % self.iter)

        training_list = copy(self.folds_list)

        validation_df = training_list.pop(self.iter)
        validation_df.reset_index(inplace=True, drop=True)
        training_df = pd.concat(training_list)
        training_df.reset_index(inplace=True, drop=True)
        self.iter += 1

        training_set = deepcopy(dataset)
        training_set.subjects_df = training_df
        validation_set = deepcopy(dataset)
        validation_set.subjects_df = validation_df

        return training_set, validation_set


def weights_init(m):
    """Initialize the weights of convolutional and fully connected layers"""
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.xavier_normal_(m.weight.data)


def adjust_learning_rate(optimizer, value):
    """Divides the learning rate by the wanted value"""
    lr_0 = optimizer.param_groups[0]['lr']
    lr = lr_0 / value
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, trainloader, validloader, epochs=1000, save_interval=5, results_path=None, model_name='model', tol=0.0,
          gpu=False, lr=0.0001):
    """
    Training a model using a validation set to find the best parameters

    :param model: The neural network to train
    :param trainloader: A Dataloader wrapping the training set
    :param validloader: A Dataloader wrapping the validation set
    :param epochs: Maximal number of epochs
    :param save_interval: The number of epochs before a new tests and save
    :param results_path: the folder where the results are saved
    :param model_name: the name given to the results files
    :param tol: the tolerance allowing the model to stop learning, when the training accuracy is (100 - tol)%
    :param gpu: boolean defining the use of the gpu (if not cpu are used)
    :param lr: the learning rate used by the optimizer
    :return:
        total training time (float)
        epoch of best accuracy for the validation set (int)
        parameters of the model corresponding to the epoch of best accuracy
    """
    filename = path.join(results_path, model_name + '.tsv')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    results_df = pd.DataFrame(columns=['epoch', 'training_time', 'acc_train', 'acc_validation'])
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')

    t0 = time()
    acc_valid_max = 0
    best_epoch = 0
    best_model = copy(model)

    # The program stops when the network learnt the training data
    epoch = 0
    acc_train = 0
    while epoch < epochs and acc_train < 100 - tol:

        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            if gpu:
                inputs, labels = data['image'].cuda(), data['diagnosis'].cuda()
            else:
                inputs, labels = data['image'], data['diagnosis']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %d] loss: %f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

        print('Finished Epoch: %d' % (epoch + 1))

        if results_path is not None and epoch % save_interval == save_interval - 1:
            training_time = time() - t0
            acc_train = test(model, trainloader, gpu)
            acc_valid = test(model, validloader, gpu)
            row = np.array([epoch + 1, training_time, acc_train, acc_valid]).reshape(1, -1)

            row_df = pd.DataFrame(row, columns=['epoch', 'training_time', 'acc_train', 'acc_validation'])
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')

            if acc_valid > acc_valid_max:
                acc_valid_max = copy(acc_valid)
                best_epoch = copy(epoch)
                best_model = deepcopy(model)

        epoch += 1

    return {'training_time': time() - t0,
            'best_epoch': best_epoch,
            'best_model': best_model,
            'acc_valid_max': acc_valid_max}


def test(model, dataloader, gpu=False):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a dataloader wrapping a dataset
    :param gpu: if True a gpu is used
    :return: balanced accuracy of the model (float)
    """

    # correct = 0
    # total = 0
    predicted_list = []
    truth_list = []

    # The test must not interact with the learning
    with torch.no_grad():
        for sample in dataloader:
            if gpu:
                images, diagnoses = sample['image'].cuda(), sample['diagnosis'].cuda()
            else:
                images, diagnoses = sample['image'], sample['diagnosis']
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_list = predicted_list + predicted.tolist()
            truth_list = truth_list + diagnoses.tolist()
            # total += diagnoses.size(0)
            # correct += (predicted == diagnoses).sum().item()

    # Computation of the balanced accuracy
    component = len(np.unique(truth_list))

    cluster_diagnosis_prop = np.zeros(shape=(component, component))
    for i, predicted in enumerate(predicted_list):
        truth = truth_list[i]
        cluster_diagnosis_prop[predicted, truth] += 1

    acc = 0
    diags_represented = []
    for i in range(component):
        diag_represented = np.argmax(cluster_diagnosis_prop[i])
        acc += cluster_diagnosis_prop[i, diag_represented] / np.sum(cluster_diagnosis_prop.T[diag_represented])
        diags_represented.append(diag_represented)

    acc = acc * 100 / component
    print('Accuracy of diagnosis: ' + str(acc))

    return acc


def cross_validation(model, dataset, folds=10, batch_size=4, **train_args):
    from torch.utils.data import DataLoader

    cross_val = CrossValidationSplit(dataset, cv=folds, stratified=True)
    acc_valid_max = 0
    best_parameters = None
    results_path = train_args['results_path']
    t0 = time()

    for i in range(folds):
        print('Fold ' + str(i + 1))
        train_args['results_path'] = path.join(results_path, 'fold-' + str(i + 1))
        if not path.exists(train_args['results_path']):
            os.makedirs(train_args['results_path'])

        trainset, validset = cross_val(dataset)

        print('Length training set', len(trainset))
        print('Length validation set', len(validset))

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)

        model.apply(weights_init)
        parameters_found = train(model, trainloader, validloader, **train_args)
        if parameters_found['acc_valid_max'] > acc_valid_max:
            best_parameters = parameters_found
            acc_valid_max = parameters_found['acc_valid_max']
            best_parameters['fold'] = i + 1

    best_parameters['training_time'] = time() - t0

    return best_parameters
