import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
import os
import pandas as pd
import torch.nn.init as init
import torch.optim as optim
from time import time
from copy import copy, deepcopy
import numpy as np



def weights_init(m):
    """Initialize the weights of convolutional and fully connected layers"""
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.xavier_normal_(m.weight.data)




class VGG(nn.Module):
    def __init__(self, n_classes=2):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 32*2, 3, padding=1)
        self.conv2 = nn.Conv3d(32*2, 64*2, 3, padding=1)
        self.conv3 = nn.Conv3d(64*2, 128*2, 3, padding=1)
        self.conv4 = nn.Conv3d(128*2, 128*2, 3, padding=1)
        self.conv5 = nn.Conv3d(128*2, 256*2, 3, padding=1)
        self.conv6 = nn.Conv3d(256*2, 256*2, 3, padding=1)
        self.conv8 = nn.Conv3d(256*2, 256*2, 3, padding=1)
        self.conv9 = nn.Conv3d(256*2, 256*2, 3, padding=1)
        self.conv7 = nn.Conv3d(256*2, 32, 1)
        self.fc1 = nn.Linear(32 * 3 * 4 * 3, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(x)

        x = F.relu(self.conv7(x))
        x = x.view(-1, 32 * 3 * 4 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x






def test(model, dataloader, device, verbose=True):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a dataloader wrapping a dataset
    :param gpu: if True a gpu is used
    :return: balanced accuracy of the model (float)
    """

    predicted_list = []
    truth_list = []
    all_prediction_scores = []
    
    # The test must not interact with the learning
    with torch.no_grad():
        for step, sample in enumerate(dataloader):
            images, diagnoses = sample['image'].cuda(non_blocking=True), sample['diagnosis_after_12_months'].cuda(non_blocking=True)
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



    
    
####################################################################################

def run_DDP(gpu,nr,gpus, world_size, model, optimizer, device, folds=10, batch_size=4,  epochs=100, results_path='scratch/di2078/results', model_name='model'):
    from torch.utils.data import DataLoader
    from data import BidsMriBrainDataset, ToTensor, GaussianSmoothing
    from training_functions import run
    import torchvision
    import torch.multiprocessing as mp
    import torch.distributed as dist
    import time

    
    ############################################################
    rank = nr * gpus + gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
    backend='nccl',
   init_method='env://',
    world_size=world_size,
    rank=rank
    )
    ############################################################
    
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    composed = torchvision.transforms.Compose([GaussianSmoothing(0), ToTensor(True)])
    train_path='/scratch/yx2105/shared/MLH/data/train.csv'
    test_path='/scratch/yx2105/shared/MLH/data/test.csv'
    valid_path='/scratch/yx2105/shared/MLH/data/val.csv'
    sigma = 0

    trainset = BidsMriBrainDataset(train_path, transform=composed)
    testset = BidsMriBrainDataset(test_path, transform=composed)
    validset = BidsMriBrainDataset(valid_path, transform=composed)
    batch_size=4
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,num_replicas=world_size,rank=rank)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset,num_replicas=world_size,rank=rank)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=train_sampler)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        validset,num_replicas=world_size,rank=rank)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=train_sampler)
    ############################################################
    #results_path = train_args['results_path']
    t0 = time.time()

    print('Length training set', len(trainset))
    print('Length validation set', len(validset))
    print('Length test set', len(testset))
    
    print('Loading complete.')
    
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
    model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu])
    filename = path.join(results_path, 'performance.csv')
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    results_df = pd.DataFrame(columns=['epoch', 'training_time', 'loss_train', 'acc_train', 'acc_validation'])
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')
    #changed acc_valid to acc_train

    acc_valid_max = 0
    best_epoch = 0
    best_model = copy(model)
    print('Before training epoch')
    # The program stops when the network learnt the training data
    epoch = 0
    acc_valid = 0
    while epoch < epochs: #and acc_train < 100 - tol:
        loss_train = []
        predicted_list = []
        truth_list = []
        running_loss = 0
        for i, data in enumerate(trainloader, 0):


            
            #inputs, labels = data['image'].to(device), data['diagnosis_after_12_months'].to(device)
            inputs, labels = data['image'].cuda(non_blocking=True), data['diagnosis_after_12_months'].cuda(non_blocking=True)
            optimizer.zero_grad()
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
        save_interval = 5
        # save performance
        if results_path is not None and epoch % save_interval == save_interval - 1:
            # compute train acc
            acc_train = compute_balanced_accuracy(predicted_list, truth_list)
            print('Training ACC: {}'.format(acc_train))

            training_time = time() - t0
            acc_valid, all_prediction_scores = test(model, validloader, device=device)

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

    parameters_found = {'training_time': time() - t0,
            'best_epoch': best_epoch,
            'best_model': best_model,
            'acc_valid_max': acc_valid_max
           }

    #parameters_found = train_DDP(gpu, model, trainloader, validloader, optimizer, device, epochs=1000, save_interval=1, results_path=results_path, model_name='model', tol=0.0)
    
####################################################################################
if __name__ == '__main__':
    from data import BidsMriBrainDataset, ToTensor, GaussianSmoothing
    from training_functions import run
    import torchvision
    import argparse

    parser = argparse.ArgumentParser()

    # DDP
    parser.add_argument("--local_rank", dest = 'local_rank',type=int, default=0)
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    #####################################################
    parser.add_argument("results_path", type=str,
                        help="where the outputs are stored")

    # Network structure
    parser.add_argument("--classifier", type=str, default='basic',
                        help='classifier selected')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of images in batch')
    parser.add_argument("--name", type=str, default='network',
                        help="name given to the outputs and checkpoints of the parameters")
    parser.add_argument("-save", "--save_interval", type=int, default=1,
                        help="the number of epochs done between the tests and saving")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1.0,
                        help='the learning rate of the optimizer (*0.00005)')

    # Managing device
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Uses gpu instead of cpu if cuda is available')
    parser.add_argument('--on_cluster', action='store_true', default=False,
                        help='to work on the cluster of the ICM')

    args = parser.parse_args()


    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    lr = args.learning_rate * 0.00005
    results_path = path.join(args.results_path, args.name)
    if not path.exists(results_path):
        os.makedirs(results_path)
        
    #composed = torchvision.transforms.Compose([GaussianSmoothing(sigma=args.sigma), ToTensor(gpu=args.gpu)])
    #train_path='/scratch/yx2105/shared/MLH/data/train.csv'
    #test_path='/scratch/yx2105/shared/MLH/data/test.csv'
    #valid_path='/scratch/yx2105/shared/MLH/data/val.csv'
    #sigma = 0

    #trainset = BidsMriBrainDataset(train_path, transform=composed)
    #testset = BidsMriBrainDataset(test_path, transform=composed)
    #validset = BidsMriBrainDataset(valid_path, transform=composed)


    if args.classifier == 'vgg':
        classifier = VGG(n_classes=args.n_classes).to(device=device)
    else:
        raise ValueError('Unknown classifier')

    # Initialization
    classifier.apply(weights_init)
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, classifier.parameters()), lr=lr,weight_decay=1e-4)

    # Training
    #best_params = run(classifier, trainset, validset, testset, optimizer, device=device, batch_size=args.batch_size, folds=args.cross_validation, epochs=args.epochs, results_path=results_path, model_name=args.name,
                                   #save_interval=args.save_interval)
     
    
    #best_params = run_DDP(classifier, optimizer, device=device, batch_size=args.batch_size, folds=10, epochs=100, results_path=results_path, model_name='model')
    world_size = args.gpus * args.nodes
    torch.multiprocessing.spawn( run_DDP, args=(args.nr,args.gpus,world_size, classifier, optimizer, device, args.batch_size, 10, 100, results_path, 'model'), nprocs=args.gpus)
    



