import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from seq_dataset_torch import SeqDataSet
from torch_lib import mask_padding
from toolkit_torch import ReadDataSet
from toolkit_torch import Statistic
from network_torch import *

import time
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


start = time.time()
'''
  HyperParameter
'''
dataset_dir = 'DataSet/SITA/'
val_dataset_dir = 'DataSet/SITA_EX1/'

kernel_width = 5
c = 256
amino_dim = 30
batch_size = 4
stage_depth = 2
dropout = 0.5
softmax_thr = 0.2
epoch = 1
learning_rate = 0.0001
softmax_thr = 0.5
threshold = softmax_thr - 0.5
test = True

'''
  평가지표
'''
train_acc = []
train_loss = []
val_acc = []
val_loss = []
val_pre = []
val_rec = []
val_AUC = []

'''
  Loss and Optimizer
'''
model = dcs_si(kernel_width, c, amino_dim, batch_size, stage_depth, dropout).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


for i in range(epoch):
    dataset = ReadDataSet(dataset_dir, 'data.feat', 'data.lab', SeqDataSet, 'SITA')
    dataset.SetSignal()
    model.train()

    all_train_acc = 0
    all_train_loss = 0

    while dataset.avail_num != 0:
        batch_x, batch_y, lens_x, lens_y = dataset.NextRestrictedPaddingBatch(batch_size)

        batch_x = torch.tensor(batch_x).to(device)
        batch_y = torch.tensor(batch_y).to(device)
        lens_x = torch.tensor(lens_x).to(device)
        lens_y = torch.tensor(lens_y).to(device)

        results = model(batch_x)
        # results = results.reshape([batch_size, -1, 2])
        results = results.squeeze().transpose(1,2)
        mask_results = mask_padding(results, lens_y, batch_size)
        soft_results = F.softmax(mask_results, dim = 1)
        thr_results = soft_results + torch.tensor([threshold, -threshold]).to(device)

        mask_batch_y = mask_padding(batch_y, lens_y, batch_size)

        prediction, correctness, train_accuracy = final_results(thr_results, batch_y, lens_y, batch_size, softmax_thr)
        print('len :', prediction.size()[0], '  sum :', sum(prediction))

        loss = criterion(thr_results, mask_batch_y)

        all_train_acc += train_accuracy
        all_train_loss += loss.data.to('cpu')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc.append(float(all_train_acc / (dataset.number // batch_size + 1)))
    train_loss.append(float(all_train_loss / (dataset.number // batch_size + 1)))

    if test:

        all_active_prob = []
        all_labels = []
        all_val_acc = 0
        all_val_loss = 0
        all_val_pre = 0
        all_val_rec = 0

        model.eval()
        val_dataset = ReadDataSet(val_dataset_dir, 'data.feat', 'data.lab', SeqDataSet, 'SITA_EX1')
        val_dataset.SetSignal()

        with torch.no_grad():
            while val_dataset.avail_num != 0:
                batch_x, batch_y, lens_x, lens_y = val_dataset.NextRestrictedPaddingBatch(batch_size)

                batch_x = torch.tensor(batch_x).to(device)
                batch_y = torch.tensor(batch_y).to(device)
                lens_x = torch.tensor(lens_x).to(device)
                lens_y = torch.tensor(lens_y).to(device)

                results = model(batch_x)

                results = results.squeeze().transpose(1, 2)
                mask_results = mask_padding(results, lens_y, batch_size)
                soft_results = F.softmax(mask_results, dim=1)
                thr_results = soft_results + torch.tensor([threshold, -threshold]).to(device)

                mask_batch_y = mask_padding(batch_y, lens_y, batch_size)

                prediction, correctness, val_accuracy = final_results(thr_results, batch_y, lens_y, batch_size, softmax_thr)

                loss = criterion(thr_results, mask_batch_y)

                tp, tn, fp, fn = Statistic(prediction, correctness)
                precision = tp / (tp + fp + 0.001)
                recall = tp / (tp + fn + 0.001)

                all_val_acc += float(val_accuracy)
                all_val_loss += float(loss.data)
                all_val_pre += precision
                all_val_rec += recall

                active_prob = []
                labels = []
                for i in range(thr_results.size()[0]):
                    active_prob.append(float(thr_results[i][1]))
                    if mask_batch_y[i][1] == 1:
                        labels.append(1.)
                    else:
                        labels.append(0.)

                all_active_prob += active_prob
                all_labels += labels

        val_acc.append(all_val_acc / (val_dataset.number // batch_size + 1))
        val_loss.append(all_val_loss / (val_dataset.number // batch_size + 1))
        val_pre.append(all_val_pre / (val_dataset.number // batch_size + 1))
        val_rec.append(all_val_rec / (val_dataset.number // batch_size + 1))
        val_AUC.append(metrics.roc_auc_score(all_labels, all_active_prob))


print(time.time() - start, '\n')

print('finished!!')