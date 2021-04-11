
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder


import argparse
from tqdm import tqdm
from loguru import logger



def lr_schedule_func(epoch):
    if epoch < 150:
        return 0.1
    elif epoch < 250:
        return 0.01
    else:
        return 0.001 

def train(model, optimizer, dataloader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = correct/total
    running_loss = train_loss/(batch_idx+1)

    return train_acc, running_loss

def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_num += 1 
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct/total
    running_loss = test_loss/len(dataloader)
    return test_acc,running_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch Imagenette Training')
    parser.add_argument('--device', default='0,1,2,3',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', '-w', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--output_dir', default='./checkpoint' ,type=str)
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    batch_size = args.batch_size
    output_dir = args.output_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy

    # Data
    logger.info('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4625, 0.4580, 0.4295), (0.2452, 0.2390, 0.2469))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.4625, 0.4580, 0.4295), (0.2452, 0.2390, 0.2469))
    ])

    trainset = ImageFolder('./data/imagenette2/train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = ImageFolder('./data/imagenette2/val/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)


    # Model
    logger.info('==> Building model..')
    model = models.resnet18(num_classes=10)
    #model = models.resnet152(num_classes=10)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info("Let's use " + str(len(args.device.split(','))) + " GPUs!")
        model = DataParallel(model, device_ids=[
                                int(i) for i in args.device.split(',')])

    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_schedule_func )

    best_acc = 0
    training_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(350):
        logger.info("Epoch {} started".format(epoch))

        train_acc,training_loss = train(model, optimizer, trainloader)
        train_acc_list.append(train_acc)
        training_loss_list.append(training_loss)
        logger.info("train acc = {:.4f}, training loss = {:.4f}".format(train_acc, training_loss))

        test_acc, test_loss = test(model, testloader)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        logger.info("test acc = {:.4f}, test loss = {:.4f}".format(test_acc, test_loss))

        if test_acc > best_acc:
            best_acc = test_acc
            logger.info("best acc improved to {:.4f}".format(best_acc))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '{}/model.pt'.format(output_dir))
            logger.info("model saved to {}/model.pt".format(output_dir))

        scheduler.step()

        logger.info("Epoch {} ended, best acc = {:.4f}".format(epoch, best_acc))
    
    
    logger.info("training loss list {}".format(training_loss_list))
    logger.info("test loss list {}".format(test_loss_list))
    logger.info("training acc list {}".format(train_acc_list))
    logger.info("test acc list {}".format(test_acc_list))
    logger.info("Training finished, best_acc = {:.4f}".format(best_acc))

        

if __name__ == '__main__':
    main()


