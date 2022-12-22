import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time
from os import path
from copy import copy, deepcopy
import pandas as pd
import numpy as np
import torch.nn.init as init
import os
from data import collate_func_img
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

class CrossValidationSplit:
    """A class to create training and validation sets for a k-fold cross validation"""

    def __init__(self, dataset, cv=5, stratified=False, shuffle_diagnosis=False, val_prop=0.10):
        """
        :param dataset: The dataset to use for the cross validation
        :param cv: The number of folds to create
        :param stratified: Boolean to choose if we have the same repartition of diagnosis in each fold
        """

        self.stratified = stratified
        subjects_df = dataset.subjects_df

        if type(cv) is int and cv > 1:
            if stratified:
                n_diagnosis = len(dataset.diagnosis_code)
                cohorts = np.unique(dataset.subjects_df.cohort.values)

                preconcat_list = [[] for i in range(cv)]
                for cohort in cohorts:
                    for diagnosis in range(n_diagnosis):
                        diagnosis_key = list(dataset.diagnosis_code)[diagnosis]
                        diagnosis_df = subjects_df[(subjects_df.diagnosis == diagnosis_key) &
                                                   (subjects_df.cohort == cohort)]

                        if shuffle_diagnosis:
                            diagnosis_df = diagnosis_df.sample(frac=1)
                            diagnosis_df.reset_index(drop=True, inplace=True)

                        for fold in range(cv):
                            preconcat_list[fold].append(diagnosis_df.iloc[int(fold * len(diagnosis_df) / cv):int((fold + 1) * len(diagnosis_df) / cv):])

                folds_list = []

                for fold in range(cv):
                    fold_df = pd.concat(preconcat_list[fold])
                    folds_list.append(fold_df)

            else:
                folds_list = []
                for fold in range(cv):
                    folds_list.append(subjects_df.iloc[int(fold * len(subjects_df) / cv):int((fold + 1) * len(subjects_df) / cv):])

            self.cv = cv

        elif type(cv) is float and 0 < cv < 1:
            if stratified:
                n_diagnosis = len(dataset.diagnosis_code)
                cohorts = np.unique(dataset.subjects_df.cohort.values)

                train_list = []
                validation_list = []
                test_list = []
                for cohort in cohorts:
                    for diagnosis in range(n_diagnosis):
                        diagnosis_key = list(dataset.diagnosis_code)[diagnosis]
                        diagnosis_df = subjects_df[(subjects_df.diagnosis == diagnosis_key) &
                                                   (subjects_df.cohort == cohort)]
                        if shuffle_diagnosis:
                            diagnosis_df = diagnosis_df.sample(frac=1)
                            diagnosis_df.reset_index(drop=True, inplace=True)

                        train_list.append(diagnosis_df.iloc[:int(len(diagnosis_df) * cv * (1-val_prop)):])
                        validation_list.append(diagnosis_df.iloc[int(len(diagnosis_df) * cv * (1-val_prop)):
                                                                 int(len(diagnosis_df) * cv):])
                        test_list.append(diagnosis_df.iloc[int(len(diagnosis_df) * cv)::])
                train_df = pd.concat(train_list)
                validation_df = pd.concat(validation_list)
                test_df = pd.concat(test_list)
                folds_list = [validation_df, train_df, test_df]

            else:
                train_df = subjects_df.iloc[:int(len(subjects_df) * cv * (1-val_prop)):]
                validation_df = subjects_df.iloc[int(len(subjects_df) * cv * (1-val_prop)):
                                                 int(len(subjects_df) * cv):]
                test_df = subjects_df.iloc[int(len(subjects_df) * cv)::]
                folds_list = [validation_df, train_df, test_df]

            self.cv = 1

        self.folds_list = folds_list
        self.iter = 0

    def __call__(self, dataset):
        """
        Calling creates a new training set and validation set depending on the number of times the function was called

        :param dataset: A dataset from data_loader.py
        :return: training set, validation set
        """
        if self.iter >= self.cv:
            raise ValueError('The function was already called %i time(s)' % self.iter)

        training_list = copy(self.folds_list)

        validation_df = training_list.pop(self.iter)
        validation_df.reset_index(inplace=True, drop=True)
        test_df = training_list.pop(self.iter - 1)
        test_df.reset_index(inplace=True, drop=True)
        training_df = pd.concat(training_list)
        training_df.reset_index(inplace=True, drop=True)
        self.iter += 1

        training_set = deepcopy(dataset)
        training_set.subjects_df = training_df
        validation_set = deepcopy(dataset)
        validation_set.subjects_df = validation_df
        test_set = deepcopy(dataset)
        test_set.subjects_df = test_df

        return training_set, validation_set, test_set


def adjust_learning_rate(optimizer, value):
    """Divides the learning rate by the wanted value"""
    lr_0 = optimizer.param_groups[0]['lr']
    lr = lr_0 / value
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, trainloader, validloader, optimizer, device, epochs=1000, save_interval=1, results_path=None, model_name='model', tol=0.0, classifier='vgg', model_path = 'none', freeze=False):
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
    if freeze:
        for name, p in model.named_parameters():
            p.requires_grad = False
            
        linear_model = nn.Sequential(nn.Linear(32 * 3 * 4 * 3, 128),
                                    nn.Linear(128, 3)).to(device)
 


    filename = path.join(results_path, 'performance.csv')

    criterion = nn.CrossEntropyLoss()

    results_df = pd.DataFrame(columns=['epoch', 'training_time', 'loss_train', 'acc_train', 'acc_validation'])
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')
    #changed acc_valid to acc_train
    t0 = time()
    acc_valid_max = 0
    best_epoch = 0
    best_model = copy(model)
    print('Before training epoch')
    # The program stops when the network learnt the training data
    epoch = 0
    acc_valid = 0
    model.train()
    while epoch < epochs:# and acc_train < 100 - tol:
        loss_train = []
        predicted_list = []
        truth_list = []
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['image'].to(device), data['diagnosis_after_12_months'].to(device)      
            optimizer.zero_grad()
            if 'joint' in classifier:
                if freeze:
                    img_features = model.classifier(inputs)
                    outputs = linear_model(img_features)
                else:
                    tab_inputs = data['tab_data'].to(device)
                    outputs = model(inputs, tab_inputs)
                
            else:
                if freeze:
                    img_features = model.feature_extractor(inputs)
                    outputs = linear_model(img_features)
                else:
                    outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()    

            loss_train.append(loss.item())

            # save for compute train acc
            preds_score = F.softmax(outputs, 1)
            _, predicted = torch.max(preds_score, 1)
            predicted_list = predicted_list + predicted.tolist()
            truth_list = truth_list + labels.tolist()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 10 mini-batches
                print('Training epoch %d : step %d / %d loss: %f' %
                      (epoch + 1,  i + 1, len(trainloader),running_loss))
                running_loss = 0.0
        
        print('Finished Epoch: %d' % (epoch + 1))

        # save performance
        if results_path is not None and epoch % save_interval == save_interval - 1:
            # compute train acc
            acc_train = compute_balanced_accuracy(predicted_list, truth_list)
            print('Training ACC: {}'.format(acc_train))

            training_time = time() - t0
            acc_valid, all_prediction_scores = test(model, validloader, device=device, classifier=classifier, freeze = freeze)

            row = np.array([epoch + 1, training_time, np.mean(loss_train) , acc_train ,acc_valid
                           ]).reshape(1, -1)

            row_df = pd.DataFrame(row, columns=['epoch', 'training_time', 'loss_train', 'acc_train', 'acc_validation'])
                                               
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')

            if acc_valid > acc_valid_max:
                acc_valid_max = copy(acc_valid)
                best_epoch = copy(epoch)
                best_model = deepcopy(model)

                torch.save(model.state_dict(), os.path.join(results_path,"checkpoint_best_val.ckpt"))
                np.save(os.path.join(results_path,"predictions_best_val"), all_prediction_scores)
        epoch += 1

    return {'training_time': time() - t0,
            'best_epoch': best_epoch,
            'best_model': best_model,
            'acc_valid_max': acc_valid_max
           }


def test(model, dataloader, device, verbose=True, classifier='vgg', freeze =False):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a dataloader wrapping a dataset
    :param gpu: if True a gpu is used
    :return: balanced accuracy of the model (float)
    """
    model.eval()
    predicted_list = []
    truth_list = []
    all_prediction_scores = []
    
    # The test must not interact with the learning
    with torch.no_grad():
        for step, sample in enumerate(dataloader):
            images, diagnoses = sample['image'].to(device), sample['diagnosis_after_12_months'].to(device)
            if 'joint' in classifier:
                if freeze:
                    img_features = model.classifier(images)
                    outputs = linear_model(img_features)
                else:
                    tab_inputs = sample['tab_data'].to(device)
                    outputs = model(images, tab_inputs)
            else:
                if freeze:
                    img_features = model.feature_extractor(images)
                    outputs = linear_model(img_features)
                else:
                    outputs = model(images)

            # save for compute train acc
            pred_scores = F.softmax(outputs,1)

            all_prediction_scores.append(pred_scores)
            _, predicted = torch.max(pred_scores, 1)
            predicted_list = predicted_list + predicted.tolist()
            truth_list = truth_list + diagnoses.tolist()
            print('Step {} / total {} processed'.format(step, len(dataloader)))

    acc = compute_balanced_accuracy(predicted_list, truth_list)

    if verbose:
        print('Validation ACC: ' + str(acc))

    all_prediction_scores = torch.cat(all_prediction_scores,0).cpu().data.numpy()

    return acc, all_prediction_scores

def compute_balanced_accuracy(predicted_list, truth_list):
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

    return acc 

def run(model, trainset, validset, testset, optimizer, device, batch_size=4, phase = 'training', **train_args):
    from torch.utils.data import DataLoader


    results_path = train_args['results_path']
    t0 = time()

    print('Length training set', len(trainset))
    print('Length validation set', len(validset))
    print('Length test set', len(testset))
    #changed num_workers from 4 to batch_size
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)# collate_fn=collate_func_img, pin_memory=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)# collate_fn=collate_func_img, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)# collate_fn=collate_func_img, pin_memory=True)
    
    print('Loading complete.')
    
    if phase == 'training':
        if train_args['model_path'] != 'none':
            
            new_weights =  OrderedDict()
            weights = torch.load(train_args['model_path'])
            for i, name in enumerate(weights):
                new_weights[name.replace('module.','')] = weights[name]

            model.load_state_dict(new_weights)
            
        parameters_found = train(model, trainloader, validloader, optimizer, device, **train_args)

    elif phase == 'inference':
        weights = torch.load(train_args['model_path'])
        model.load_state_dict(weights)
        acc_test, all_prediction_scores = test(model, testloader, device=device, classifier = train_args['classifier'])

        np.save(os.path.join(results_path,"predictions_best_test"), all_prediction_scores)
        print('Accuracy on test set: %.2f %% \n' % acc_test)
        

    #acc_test = test(parameters_found['best_model'], testloader, train_args['gpu'], verbose=False)
    #acc_valid = test(parameters_found['best_model'], validloader, train_args['gpu'], verbose=False)
    #acc_train = test(parameters_found['best_model'], trainloader, device=device, verbose=False)

    # text_file = open(path.join(train_args['results_path'], 'fold_output.txt'), 'w')
    #     #text_file.write('Fold: %i \n' % (i + 1))
    # text_file.write('Best epoch: %i \n' % (parameters_found['best_epoch'] + 1))
    # text_file.write('Time of training: %d s \n' % parameters_found['training_time'])
    # text_file.write('Accuracy on training set: %.2f %% \n' % acc_train)
    # #text_file.write('Accuracy on validation set: %.2f %% \n' % acc_valid)
    # #text_file.write('Accuracy on test set: %.2f %% \n' % acc_test)
    # text_file.close()

    # print('Accuracy of the network on the %i train images: %.2f %%' % (len(trainset), acc_train))
    # #print('Accuracy of the network on the %i validation images: %.2f %%' % (len(validset), acc_valid))
    # #print('Accuracy of the network on the %i test images: %.2f %%' % (len(testset), acc_test))

    #     #accuracies[i] = acc_test

    # training_time = time() - t0
    # text_file = open(path.join(results_path, 'model_output.txt'), 'w')
    # text_file.write('Time of training: %d s \n' % training_time)
    # #text_file.write('Mean test accuracy: %.2f %% \n' % np.mean(accuracies))
    # text_file.close()