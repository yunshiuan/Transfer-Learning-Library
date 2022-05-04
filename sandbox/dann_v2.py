"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import os.path as osp
import os
import shutil
import argparse
import warnings
from tkinter import ARC
import time
import random
import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")
sys.path.insert(0, "scripts/lib")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime. But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
    cudnn.benchmark = True

    # ------------------
    # Data loading
    # - 2 domains (source, target) x 3 partitions (train, val, test)
    # - note: the difference between train_transform and val_transform is `RandomResizedCrop()` and `RandomHorizontalFlip()`.
    # ------------------
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    # - len(source_train_dataset) = 1690
    # - len(source_val_dataset) = 563
    # - len(source_test_dataset) = 564
    source_train_dataset, source_val_dataset, source_test_dataset, num_classes, args.class_names = \
        utils.get_dataset_with_partition(dataset_name=args.data, root=args.root,
                                         domain_train=args.source_train, domain_val=args.source_val, domain_test=args.source_test,
                                         train_transform=train_transform,
                                         val_transform=val_transform)

    target_train_dataset, target_val_dataset, target_test_dataset, num_classes, args.class_names = \
        utils.get_dataset_with_partition(dataset_name=args.data, root=args.root,
                                         domain_train=args.target_train, domain_val=args.target_val, domain_test=args.target_test,
                                         train_transform=train_transform,
                                         val_transform=val_transform)

    # - 'data/office31/image_list/amazon_train.txt'
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    # - 'data/office31/image_list/webcam_train.txt'
    target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    # 'data/office31/image_list/amazon_val.txt'
    source_val_loader = DataLoader(
        source_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # 'data/office31/image_list/webcam_val.txt'
    target_val_loader = DataLoader(
        target_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # 'data/office31/image_list/amazon_test.txt'
    source_test_loader = DataLoader(
        source_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # 'data/office31/image_list/webcam_test.txt'
    target_test_loader = DataLoader(
        target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    dict_loader = {
        "source_train_loader": source_train_loader,
        "target_train_loader": target_train_loader,
        "source_val_loader": source_val_loader,
        "target_val_loader": target_val_loader,
        "source_test_loader": source_test_loader,
        "target_test_loader": target_test_loader
    }

    # data iterators that will never stop producing data
    source_train_iter = ForeverDataIterator(source_train_loader)
    target_train_iter = ForeverDataIterator(target_train_loader)
    # source_val_iter = ForeverDataIterator(source_val_loader)
    # target_val_iter = ForeverDataIterator(target_val_loader)
    # source_test_iter = ForeverDataIterator(source_test_loader)
    # target_test_iter = ForeverDataIterator(target_test_loader)

    # ------------------
    # create the DANN model
    # - classifier (ImageClassifier):
    # -- backbone (feature extractor) -> pool_layer -> bottleneck -> (domain-invariant feature, dim = args.bottleneck_dim = 256) -> head (label predictor)
    # --- backbone: 'resnet50':
    #   ---- input.shape = torch.Size([64, 3, 224, 224])
    #   ---- output.shape: torch.Size([64, 2048, 7, 7])
    #
    # --- pool_layer:
    # Sequential(
    #   (0): AdaptiveAvgPool2d(output_size=(1, 1))
    #   (1): Flatten(start_dim=1, end_dim=-1)
    # )
    #   ---- input.shape = torch.Size([64, 2048, 7, 7])
    #   ---- output.shape = torch.Size([64, 2048])
    #
    # --- bottleneck:
    #  Sequential(
    #   (0): Linear(in_features=2048, out_features=256, bias=True)
    #   (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #   (2): ReLU()
    # )
    #   ---- input.shape = torch.Size([64, 2048])
    #   ---- output.shape [the domain-invariant feature] = torch.Size([64, 256])
    #
    # --- head:
    #  Linear(in_features=256, out_features=31, bias=True)
    #   ---- input.shape = torch.Size([64, 256])
    #   ---- output.shape = torch.Size([64, 31]) # 31 = 31 object categories in three domains: Amazon, DSLR and Webcam
    #
    #
    # - domain discri:
    # DomainDiscriminator(
    #   (0): Linear(in_features=256, out_features=1024, bias=True)
    #   (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #   (2): ReLU()
    #   (3): Linear(in_features=1024, out_features=1024, bias=True)
    #   (4): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #   (5): ReLU()
    #   (6): Sequential(
    #         (0): Linear(in_features=1024, out_features=1, bias=True)
    #       (1): Sigmoid()
    #   )
    # )
    #   -- input.shape = torch.Size([64, 256])
    #   -- output.shape = torch.Size([64, 1])
    # ------------------
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    # - if no_pool=False, then the pooling layer is set to AdaptiveAvgPool2d() (see classifier.py)
    pool_layer = nn.Identity() if args.no_pool else None

    # - args.bottleneck_dim = 256
    # issubclass(ImageClassifier, nn.Module) -> True
    # isinstance(classifier, nn.Module) -> True
    classifier = ImageClassifier(backbone, num_classes,
                                 bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    # batch = next(source_train_iter)
    # x=batch[0]
    # x=x.to(device)
    # y1=backbone(x)
    # y2=classifier(x)

    domain_discri = DomainDiscriminator(
        in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # ------------------
    # define optimizer and lr scheduler
    # ------------------
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # ------------------
    # define the adversarial loss function
    #
    # DomainAdversarialLoss(
    # (grl): WarmStartGradientReverseLayer()
    # (domain_discriminator): DomainDiscriminator(
    #     (0): Linear(in_features=256, out_features=1024, bias=True)
    #     (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (2): ReLU()
    #     (3): Linear(in_features=1024, out_features=1024, bias=True)
    #     (4): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (5): ReLU()
    #     (6): Sequential(
    #     (0): Linear(in_features=1024, out_features=1, bias=True)
    #     (1): Sigmoid()
    #     )
    # )
    # )
    # ------------------
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # is test or analysis phase: resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(
            logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # # analysis the model
    # if args.phase == 'analysis':
    #     # extract features from both domains
    #     feature_extractor = nn.Sequential(
    #         classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
    #     source_feature = collect_feature(
    #         source_train_loader, feature_extractor, device)
    #     target_feature = collect_feature(
    #         target_train_loader, feature_extractor, device)
    #     # plot t-SNE
    #     tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
    #     tsne.visualize(source_feature, target_feature, tSNE_filename)
    #     print("Saving t-SNE to", tSNE_filename)
    #     # calculate A-distance, which is a measure for distribution discrepancy
    #     A_distance = a_distance.calculate(
    #         source_feature, target_feature, device)
    #     print("A-distance =", A_distance)
    #     return

    if args.phase == 'test':
        # ------------------
        # evaluate on all the data sets
        # ------------------
        for k_set in dict_loader:
            # top-1 accuracy
            acc1 = utils.validate(
                dict_loader[k_set], classifier, args, device, print_progress=False)
            print("Top-1 accuracy on {}: {}".format(k_set, acc1))
        return

    # ------------------
    # start training
    # ------------------
    if USE_WB:
        wandb_config = {
            "seed": args.seed,
            "trade_off": args.trade_off,
            "version": args.version,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "iters_per_epoch": args.iters_per_epoch,
            "dataset": DATASET,
            "lr_start": args.lr,
            "lr_decay": args.lr_decay,
            "lr_gamma": args.lr_gamma,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "workers": args.workers,
            "classifier": str(classifier),
            "domain_adv": str(domain_adv)
        }
        wandb.init(project="transfer-demo", config=wandb_config)
        wandb.run.name = '{}'.format(args.version)
        wandb.run.save()
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # ------------------
        # the learning rate at the start of this epoch
        # - lr decays throughout the time
        # ------------------
        print("--------------------------")
        print("Start of epoch [{}/{}]".format(epoch, args.epochs))
        print("lr:", lr_scheduler.get_last_lr()[0])
        print("--------------------------")
        if USE_WB:
            wandb.log(
                {
                    "lr_epoch": lr_scheduler.get_last_lr()[0]
                })
        # ------------------
        # train for one epoch (for each epoch, train for `iters-per-epoch` iterations)
        # ------------------
        train(source_train_iter, target_train_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, args)

        # ------------------
        # evaluate at this epoch on the validation set
        # - in this example, validation dataset == test dataset == train-target dataset == 'data/office31/image_list/webcam.txt'
        # - len = 795
        # ------------------
        # - the validation top-1 accuracy on label prediction
        # -- val_loader: the DataLoader for the val set
        # -- classifier: the main classifier with the label predictor at the end
        acc1 = utils.validate(source_val_loader, classifier, args, device)

        # remember best acc@1 (through all epochs) and save checkpoint
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        print("--------------------------")
        print("End of epoch [{}/{}]".format(epoch, args.epochs))
        print("current top-1 acc on source_val= {:3.1f}".format(acc1))
        print("--------------------------")
        if USE_WB:
            wandb.log(
                {
                    "epoch_top1_source_val": acc1,
                })
    print("--------------------------")
    print("End of the training")
    print("Best top-1 acc on source_val= {:3.1f}".format(best_acc1))
    print("--------------------------")
    logger.close()


def train(source_train_iter: ForeverDataIterator, target_train_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    # ------------------
    # Meters for compute and store the average and current value.
    # ------------------
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    losses_cls = AverageMeter('Loss_cls', ':6.2f')
    losses_domain = AverageMeter('Loss_domain', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time,
         losses, losses_cls, losses_domain,
         cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()

    # ------------------
    # iterate all the instances within the epoch
    # ------------------
    for i in range(args.iters_per_epoch):

        # ------------------
        # get the data ((x_s,label_s),x_t)
        #
        # x_s: the intput features of the source domain
        #   - shape = torch.Size([32, 3, 224, 224]) # batch_size, # channels, h, w
        # label_s: the input labels of the source domain
        #   - shape = 32
        # x_t: the intput features of the target domain
        #   - shape: torch.Size([32, 3, 224, 224])
        # ------------------
        # get the features and the labels in the source domain
        x_s, labels_s = next(source_train_iter)[:2]
        # get the features in the target domain
        x_t, = next(target_train_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # ------------------
        # feedforward: compute output
        # - concatenate the inputs from the source and the target domain only for convenience
        # -- treating them as separate batches
        # ------------------
        # x = (x_s, s_t)
        # - x.shape = torch.Size([64, 3, 224, 224])
        x = torch.cat((x_s, x_t), dim=0)

        # y = y_hat = (y_hat_s, y_hat_t)
        # - y.shape = torch.Size([64, 31]), 31 =  # classes
        # f = the domain-invariant feature
        # - f.shape = torch.Size([64, 256])
        y, f = model(x)

        # split y_hat and f back to source and target domain
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        # ------------------
        # compute the loss
        # - total loss = classification loss + domain adversarial loss
        # -- classification loss = cross entropy loss of (source y_hat, source y_ground_truth)
        # -- domain adversarial loss = the domain adversarial loss between (f_s, f_t)
        # ------------------
        # - classification loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        # - domain adversarial loss
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        # args.trade_off = the relative weight of the domain adversarial loss with respect to the classification loss
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        # ------------------
        # update the averahe loss, classication acc, domain accuracy
        # - for domain accuracy, the closer it's to the 'random rate', the better it is
        # ------------------
        losses.update(loss.item(), x_s.size(0))
        losses_cls.update(cls_loss.item(), x_s.size(0))
        losses_domain.update(transfer_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # ------------------
        # back-propagation
        # - compute gradient and do the SGD step
        # ------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure the elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # ------------------
        # print the acc and loss at this time
        # - e.g., Epoch: [17][ 900/1000]	Time  0.38 ( 0.38)	Data  0.11 ( 0.12)	Loss   0.76 (  0.75)	Cls Acc 100.0 (98.4)	Domain Acc 56.2 (56.4)
        # -- value of this iteration (average so far within this epoch)
        # ------------------
        if i % args.print_freq == 0:
            progress.display(i)
            if USE_WB:
                wandb.log(
                    {
                        "lr_iter": lr_scheduler.get_last_lr()[0],
                        "loss": losses.avg,
                        "loss_cls": losses_cls.avg,
                        "loss_domain": losses_domain.avg,
                        "acc_cls": cls_accs.avg,
                        "acc_domain": domain_accs.avg
                    })


if __name__ == '__main__':

    # - CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W
    # - par
    DATASET = "Office31_v2"
    VERSION = "v6"
    PHASE = 'train'
    # PHASE = 'test'
    # - the weight for the adversarial loss
    TRADEOFF = 0.  # 1.

    DOMAIN_SOURCE_TRAIN = "A_train"
    DOMAIN_SOURCE_VAL = "A_val"
    DOMAIN_SOURCE_TEST = "A_test"

    DOMAIN_TARGET_TRAIN = "W_train"
    DOMAIN_TARGET_VAL = "W_val"
    DOMAIN_TARGET_TEST = "W_test"

    NUM_EPOCHS = 20  # 20
    SEED = 1
    ARCH = "resnet50"
    ITERS_PER_EPOCH = 1000  # 1000
    USE_WB = True

    # - path
    PATH_ROOT = '/home/sean/CS769/project/Transfer-Learning-Library'
    PATH_DATA = osp.join(PATH_ROOT, 'data')
    PATH_LOG = osp.join(PATH_ROOT, 'sandbox', 'logs', 'dann', DATASET, VERSION)

    parser = argparse.ArgumentParser(
        description='DANN for Unsupervised Domain Adaptation')
    # # dataset parameters
    parser.add_argument('-root', metavar='DIR',
                        help='root path of dataset', default=PATH_DATA)
    parser.add_argument('-d', '--data', metavar='DATA', default=DATASET, choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s_train', '--source_train', help='source domain(s) train set',
                        nargs='+', default=DOMAIN_SOURCE_TRAIN)
    parser.add_argument('-t_train', '--target_train', help='target domain(s) train set',
                        nargs='+', default=DOMAIN_TARGET_TRAIN)
    parser.add_argument('-s_val', '--source_val', help='source domain(s) validation set',
                        nargs='+', default=DOMAIN_SOURCE_VAL)
    parser.add_argument('-t_val', '--target_val', help='target domain(s) validation set',
                        nargs='+', default=DOMAIN_TARGET_VAL)
    parser.add_argument('-s_test', '--source_test', help='source domain(s) test set',
                        nargs='+', default=DOMAIN_SOURCE_TEST)
    parser.add_argument('-t_test', '--target_test', help='target domain(s) test set',
                        nargs='+', default=DOMAIN_TARGET_TEST)
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default=ARCH,  # 'resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch (use the pretrained otherwise).')
    parser.add_argument('--trade-off', default=TRADEOFF, type=float,
                        help='the trade-off hyper-parameter for transfer loss (the relative weight of the domain adversarial loss with respect to the classification loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=NUM_EPOCHS, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=ITERS_PER_EPOCH, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=SEED, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default=PATH_LOG,  # 'dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default=PHASE, choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--version', default=VERSION)
    args = parser.parse_args()
    main(args)
