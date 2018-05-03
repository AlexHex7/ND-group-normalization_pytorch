import logging
import torch
import torch.utils.data as Data
import torchvision
from lib.network import Network
from torch import nn
from lib.utils import create_architecture
import config as cfg
import os

torch.manual_seed(1)

train_loss_record_file = os.path.join(cfg.loss_record_dir, 'BN_2_train.log')
test_loss_record_file = os.path.join(cfg.loss_record_dir, 'BN_2_test.log')
if os.path.exists(train_loss_record_file):
    os.remove(train_loss_record_file)
if os.path.exists(test_loss_record_file):
    os.remove(test_loss_record_file)


def calc_acc(x, y):
    x = torch.max(x, dim=-1)[1]
    accuracy = sum(x == y).float() / x.size(0)
    return accuracy


logging.getLogger().setLevel(logging.INFO)
create_architecture()

train_data = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST(root='./mnist/',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False)

train_batches = train_loader.__len__()
train_samples = train_loader.dataset.__len__()

test_batches = test_loader.__len__()
test_samples = test_loader.dataset.__len__()

print('-' * 40)
print('Train: %d batches, %d samples' % (train_batches, train_samples))
print('Test: %d batches, %d samples' % (test_batches, test_samples))
print('-' * 40 + '\n')


net = Network()
if torch.cuda.is_available():
    net.cuda(cfg.cuda_num)

opt = torch.optim.Adam(net.parameters(), lr=cfg.LR, weight_decay=cfg.weight_decay)
loss_func = nn.CrossEntropyLoss()

if cfg.load_model:
    net.load_state_dict(torch.load(cfg.model_path))

count = 0
for epoch_index in range(cfg.epoch):
    # =============================== Training =====================================
    for train_batch_index, (img_batch, label_batch) in enumerate(train_loader):
        torch.set_grad_enabled(True)
        net.train()

        count += 1
        if torch.cuda.is_available():
            img_batch = img_batch.cuda(cfg.cuda_num)
            label_batch = label_batch.cuda(cfg.cuda_num)

        predict = net(img_batch)
        acc = calc_acc(predict.cpu(), label_batch.cpu())
        loss = loss_func(predict, label_batch)

        net.zero_grad()
        loss.backward()
        opt.step()

        with open(train_loss_record_file, 'a+') as fp:
            fp.write('%d %.4f\n' % (count, loss.item()))

        # if count % 32 != 0:
        #     continue
        # ============================ Testing ====================================
        torch.set_grad_enabled(False)
        net.eval()

        total_loss = 0
        total_acc = 0
        for test_batch_index, (img_batch, label_batch) in enumerate(test_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda(cfg.cuda_num)
                label_batch = label_batch.cuda(cfg.cuda_num)

            predict = net(img_batch)
            acc = calc_acc(predict.cpu(), label_batch.cpu())
            loss = loss_func(predict, label_batch)

            total_loss += loss
            total_acc += acc

        mean_acc = total_acc / test_batches
        mean_loss = total_loss / test_batches
        logging.info('%d [Test] epoch[%d/%d] acc:%.4f loss:%.4f '
                     % (count, epoch_index, cfg.epoch, mean_acc, mean_loss.item()))

        with open(test_loss_record_file, 'a+') as fp:
            fp.write('%d %.4f\n' % (count, mean_loss.item()))

    torch.save(net.state_dict(), cfg.model_path)
