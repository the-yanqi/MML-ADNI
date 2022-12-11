import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
import os
import pandas as pd
import torch.nn.init as init
import torch.optim as optim



def weights_init(m):
    """Initialize the weights of convolutional and fully connected layers"""
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.xavier_normal_(m.weight.data)


def train(model, trainloader, validloader, epochs=1000, save_interval=5, results_path=None, model_name='model', tol=0.0,
          gpu=False, lr=0.1):
    #changed learning rate
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
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=lr)
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
            outputs = model(inputs, train=True)
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



if __name__ == '__main__':
    from data import BidsMriBrainDataset, ToTensor, GaussianSmoothing, collate_func_img
    from training_functions import run
    import torchvision
    import argparse
    from model import *
    #print("STARTING......?")
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    #parser.add_argument("train_path", type=str,
     #                   help='path to your list of subjects for training')
    parser.add_argument("results_path", type=str,
                        help="where the outputs are stored")
    #parser.add_argument("caps_path", type=str,
     #                   help="path to your caps folder")

    # Network structure
    parser.add_argument("--classifier", type=str, default='basic',
                        help='classifier selected')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes in the dataset')

    # Dataset management
    parser.add_argument('--bids', action='store_true', default=False)
    parser.add_argument('--sigma', type=float, default=0,
                        help='Size of the Gaussian smoothing kernel (preprocessing)')
    parser.add_argument('--rescale', type=str, default='crop',
                        help='Action to rescale the BIDS without deforming the images')

    # Training arguments
    parser.add_argument("-e", "--epochs", type=int, default=60,
                        help="number of loops on the whole dataset")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1.0,
                        help='the learning rate of the optimizer (*0.00005)')
    parser.add_argument('-cv', '--cross_validation', type=int, default=10,
                        help='cross validation parameter')
    parser.add_argument('--dropout', '-d', type=float, default=0.5,
                        help='Dropout rate before FC layers')
    parser.add_argument('--batch_size', '-batch', type=int, default=4,
                        help="The size of the batches to train the network")
    parser.add_argument("--model_path", type=str, default='basic',
                        help='checkpoint path')
    parser.add_argument("--phase", type=str, default='training',
                        help='experiment phase, ')

    # Managing output
    parser.add_argument("-n", "--name", type=str, default='network',
                        help="name given to the outputs and checkpoints of the parameters")
    parser.add_argument("-save", "--save_interval", type=int, default=1,
                        help="the number of epochs done between the tests and saving")

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
    #print("STARTING......")
    composed = torchvision.transforms.Compose([GaussianSmoothing(sigma=args.sigma), ToTensor(gpu=args.gpu)])
    train_path='/scratch/yx2105/shared/MLH/data/train.csv'
    test_path='/scratch/yx2105/shared/MLH/data/test.csv'
    valid_path='/scratch/yx2105/shared/MLH/data/val.csv'
    sigma = 0
    #composed = torchvision.transforms.Compose([GaussianSmoothing(sigma),])
    trainset = BidsMriBrainDataset(train_path, transform=composed)
    testset = BidsMriBrainDataset(test_path, transform=composed)
    validset = BidsMriBrainDataset(valid_path, transform=composed)

    if args.classifier == 'basic':
        classifier = DiagnosisClassifier(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'simonyan':
        classifier = SimonyanClassifier(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'simple':
        classifier = SimpleClassifier(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'basicgpu':
        classifier = BasicGPUClassifier(n_classes=args.n_classes, dropout=args.dropout).to(device=device)
    elif args.classifier == 'simpleLBP':
        classifier = SimpleLBP(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'simpleLBCNN':
        classifier = SimpleLBCNN(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'localbriefnet':
        classifier = LocalBriefNet(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'localbriefnet2':
        classifier = LocalBriefNet2(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'localbriefnet3':
        classifier = LocalBriefNet3(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'vgg':
        classifier = VGG(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'cnn':
        classifier = CNNModel(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'joint':
        classifier = joint_model(tab_in_shape = 49, enc_shape = 8, n_classes = 3, classifier='vgg').to(device=device)
    else:
        raise ValueError('Unknown classifier')

    # Initialization
    classifier.apply(weights_init)
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, classifier.parameters()), lr=lr,weight_decay=1e-4)

    # Training
    best_params = run(classifier, trainset, validset, testset, optimizer, device=device, batch_size=args.batch_size, epochs=args.epochs, phase=args.phase, results_path=results_path, model_name=args.name,
                                   save_interval=args.save_interval, classifier = args.classifier)
