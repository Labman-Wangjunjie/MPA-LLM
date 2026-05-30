
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch as t
import torch.nn as nn
from torch import optim
import pandas as pd
import os
import time
import warnings
from tqdm import tqdm
import numpy as np
from utils.metrics_anomaly import get_anomaly_metrics

warnings.filterwarnings('ignore')

class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(self, pred, true):
        x_loss = torch.mean(100 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        return x_loss


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



class Exp_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Detection, self).__init__(args)

        self.pred_len = 0
        self.seq_len = self.args.seq_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            print(f'len_vali:{len(vali_loader)}')
            for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                if self.args.data == 'net_traffic_wsdream':
                    batch_x = batch_x.permute(0, 2, 1)
                B, T, N = batch_x.shape
                mask = torch.zeros((B, T, N)).to(self.device)
                sample_rate = self.args.mask_rate
                for col in range(N):

                    random_indices = torch.randperm(T)[:int(sample_rate * T)]
                    col_mask = torch.zeros(T, device=batch_x.device)
                    col_mask[random_indices] = 1

                    mask[:, :, col] = col_mask.unsqueeze(0).expand(B, -1)

                inp = batch_x.masked_fill(mask == 0, 0)


                for col in range(N):

                    col_mask = mask[0, :, col]

                    unmasked_indices = torch.nonzero(col_mask, as_tuple=True)[0]

                    masked_indices = torch.nonzero(col_mask == 0, as_tuple=True)[0]

                    if len(masked_indices) > 0:

                        random_mapping = torch.randint(0, len(unmasked_indices), (len(masked_indices),))

                        inp[:, masked_indices, col] = batch_x[:, unmasked_indices[random_mapping], col]

                known_rate = self.args.mask_rate * 100
                target_rate = known_rate
                times = np.ceil(np.log2(100 / known_rate)).astype(int)
                outputs = inp
                for step_idx in range(6):
                    outputs = self.model(outputs, None, stage_idx=step_idx)
                    outputs = torch.where(mask == 1, batch_x, outputs)
                # outputs = torch.where(mask == 1, batch_x, outputs)
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.cpu()
                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("="*50)
        print(f"Model Parameters: Total = {self.total_params:,} | Trainable = {self.trainable_params:,}")
        print("="*50)

        scaler = torch.cuda.amp.GradScaler()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        total_train_start_time = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = nn.L1Loss()

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(f'len_train:{len(train_loader)}')
            for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
                all_loss = 0
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                if self.args.data == 'net_traffic_wsdream':
                    batch_x = batch_x.permute(0, 2, 1)
                    B, T, N = batch_x.shape

                    inp_list = []

                    inp_2 = batch_x.clone()

                    for i in range(0, T, 64):
                        inp_2[:, i:i + 63, :] = batch_x[:, i + 63:i + 64, :]
                    inp_list.append(inp_2)

                    inp_4 = batch_x.clone()

                    for i in range(0, T, 32):
                        inp_4[:, i:i + 31, :] = batch_x[:, i + 31:i + 32, :]
                    inp_list.append(inp_4)

                    inp_8 = batch_x.clone()
                    for i in range(0, 60, 10):
                        inp_8[:, i:i + 9, :] = batch_x[:, i + 9:i + 10, :]

                    inp_list.append(inp_8)

                    inp_16 = batch_x.clone()
                    for i in range(0, 60, 6):
                        inp_16[:, i:i + 5, :] = batch_x[:, i + 5:i + 6, :]

                    inp_list.append(inp_16)

                    inp_32 = batch_x.clone()
                    for i in range(0, 63, 3):
                        inp_32[:, i:i + 2, :] = batch_x[:, i + 2:i + 3, :]

                    inp_list.append(inp_32)

                    inp_64 = batch_x.clone()
                    for i in range(0, T, 2):
                        inp_64[:, i:i + 1, :] = batch_x[:, i + 1:i + 2, :]
                else:
                    B, T, N = batch_x.shape

                    inp_list = []

                    inp_2 = batch_x.clone()

                    for i in range(0, T, 50):
                        inp_2[:, i:i + 49, :] = batch_x[:, i + 49:i + 50, :]
                    inp_list.append(inp_2)

                    inp_4 = batch_x.clone()

                    for i in range(0, T, 25):
                        inp_4[:, i:i + 24, :] = batch_x[:, i + 24:i + 25, :]
                    inp_list.append(inp_4)

                    inp_8 = batch_x.clone()
                    for i in range(0, 48, 12):
                        inp_8[:, i:i + 11, :] = batch_x[:, i + 11:i + 12, :]

                    inp_list.append(inp_8)

                    inp_16 = batch_x.clone()
                    for i in range(0, 48, 6):
                        inp_16[:, i:i + 5, :] = batch_x[:, i + 5:i + 6, :]

                    inp_list.append(inp_16)

                    inp_32 = batch_x.clone()
                    for i in range(0, 48, 3):
                        inp_32[:, i:i + 2, :] = batch_x[:, i + 2:i + 3, :]

                    inp_list.append(inp_32)

                    inp_64 = batch_x.clone()
                    for i in range(0, 48, 3):
                        inp_64[:, i:i + 1, :] = batch_x[:, i + 2:i + 3, :]

                inp_list.append(inp_64)

                inp_100 = batch_x.clone()
                inp_list.append(inp_100)

                for step_idx in range(6):
                    model_optim.zero_grad()
                    outputs = self.model(inp_list[step_idx], None, stage_idx=step_idx)
                    loss = criterion(outputs, inp_list[step_idx + 1])
                    loss.backward()
                    model_optim.step()
                    all_loss += loss.item()
                train_loss.append(all_loss / 6)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, all_loss / 6))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                torch.cuda.empty_cache()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            torch.cuda.empty_cache()
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.total_train_time = time.time() - total_train_start_time
        print("="*50)
        print(f"⏱️ Total Training Time: {self.total_train_time:.4f} seconds")
        print("="*50)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        print('loading model')
        if hasattr(self.args, 'zero_shot_ckpt') and self.args.zero_shot_ckpt:
            ckpt_path = os.path.join(self.args.zero_shot_ckpt, 'checkpoint.pth')
            print(f"0-shot: {ckpt_path}")
        else:
            ckpt_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(ckpt_path))
        # self.model.load_state_dict(torch.load(ckpt_path), strict=False)

        preds = []
        trues = []
        masks = []
        labels = []

        self.model.eval()
        # with torch.no_grad():
        print(f'len_test:{len(test_loader)}')
        test_start_time = time.time()

        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(self.device)
            if self.args.data == 'net_traffic_wsdream':
                batch_x = batch_x.permute(0, 2, 1)
            B, T, N = batch_x.shape
            features = N

            mask = torch.zeros((B, T, N)).to(self.device)
            sample_rate = self.args.mask_rate
            for col in range(N):
                random_indices = torch.randperm(T)[:int(sample_rate * T)]
                col_mask = torch.zeros(T, device=batch_x.device)
                col_mask[random_indices] = 1
                mask[:, :, col] = col_mask.unsqueeze(0).expand(B, -1)

            inp = batch_x.masked_fill(mask == 0, 0)

            for col in range(N):
                col_mask = mask[0, :, col]
                unmasked_indices = torch.nonzero(col_mask, as_tuple=True)[0]
                masked_indices = torch.nonzero(col_mask == 0, as_tuple=True)[0]
                if len(masked_indices) > 0:
                    random_mapping = torch.randint(0, len(unmasked_indices), (len(masked_indices),))
                    inp[:, masked_indices, col] = batch_x[:, unmasked_indices[random_mapping], col]

            known_rate = self.args.mask_rate * 100
            target_rate = known_rate
            times = np.ceil(np.log2(100 / known_rate)).astype(int)
            outputs = inp
            for step_idx in range(6):
                outputs = self.model(outputs, None, stage_idx=step_idx)
                outputs = torch.where(mask == 1, batch_x, outputs)
            # outputs = torch.where(mask == 1, batch_x, outputs)
            outputs = outputs.detach().cpu().numpy()
            pred = outputs
            true = batch_x.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)
            masks.append(mask.detach().cpu())
            labels.append(batch_y.detach().cpu().numpy())

        test_total_time = time.time() - test_start_time

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        labels = np.concatenate(labels, 0)

        if self.args.data == 'net_traffic_wsdream':
            preds = preds.transpose(0, 2, 1)
            trues = trues.transpose(0, 2, 1)
            masks = masks.transpose(0, 2, 1)

        #3/20-add-begin
        anomaly_scores = np.abs(preds - trues)
        print(f"granularity: {self.args.granularity}")

        if self.args.granularity == 'time_step':
            labels_eval = np.max(labels, axis=-1).flatten()
            scores_eval = np.max(anomaly_scores, axis=-1).flatten()
            total_time_steps = len(labels_eval)
            anomalous_steps = np.sum(labels_eval)
            anomaly_ratio = (anomalous_steps / total_time_steps) * 100
            print(
                f" total_time_steps: {total_time_steps} | anomalous_steps: {anomalous_steps} | anomaly_ratio: {anomaly_ratio:.4f}%")

        elif self.args.granularity == 'point':
            labels_eval = labels.flatten()
            scores_eval = anomaly_scores.flatten()
            total_time_steps = len(labels_eval)
            anomalous_steps = np.sum(labels_eval)
            anomaly_ratio = (anomalous_steps / total_time_steps) * 100
            print(f" total_time_steps: {total_time_steps} | anomalous_steps: {anomalous_steps} | anomaly_ratio: {anomaly_ratio:.4f}%")
        else:
            raise ValueError("choose 'time_step' or 'point'")

        print("...")
        labels_eval = (labels_eval > 0.5).astype(int)
        target_ratio = self.args.anomaly_ratio

        auc_roc, auc_prc, best_th, raw_metrics, pa_metrics, aff_metrics = get_anomaly_metrics(
            scores_eval,
            labels_eval,
            target_anomaly_ratio=target_ratio
        )

        raw_p, raw_r, raw_f1 = raw_metrics
        pa_p, pa_r, pa_f1 = pa_metrics
        aff_p, aff_r, aff_f1 = aff_metrics

        print(f"====== Anomaly Detection Results ({self.args.granularity}) ======")
        print(f"ROC-AUC: {auc_roc:.4f}")
        print(f"PRC-AUC: {auc_prc:.4f}")
        print(f"Best Threshold: {best_th:.4f}")
        print(f"[Raw]     Precision: {raw_p:.4f}, Recall: {raw_r:.4f}, F1: {raw_f1:.4f}")
        print(f"[PA-Adj]  Precision: {pa_p:.4f}, Recall: {pa_r:.4f}, F1: {pa_f1:.4f}")
        print(f"[Aff]     Precision: {aff_p:.4f}, Recall: {aff_r:.4f}, F1: {aff_f1:.4f}")
        print("==================================================================")

        print('test shape:', preds.shape, trues.shape, labels.shape)
        nmae, nrmse, kl = metric(preds[masks == 0], trues[masks == 0])
        print('nmae:{}, nrmse:{} kl:{}'.format(nmae, nrmse, kl))

        print("=" * 50)
        print(f"total_train_time: {self.total_train_time:.4f} seconds")
        print(f"Total Testing/Inference Time: {test_total_time:.4f} seconds")
        print(f"Total params: {self.total_params:.4f}, Trainable:{self.trainable_params:.4f} ")
        print("=" * 50)

        f = open("result_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write(f"Anomaly Ratio Lock: {target_ratio}%\n")
        f.write(f"ROC-AUC: {auc_roc:.4f}\n")
        f.write(f"[Raw]     Precision: {raw_p:.4f}, Recall: {raw_r:.4f}, F1: {raw_f1:.4f}\n")
        f.write(f"[PA-Adj]  Precision: {pa_p:.4f}, Recall: {pa_r:.4f}, F1: {pa_f1:.4f}\n")
        f.write(f"[Aff]     Precision: {aff_p:.4f}, Recall: {aff_r:.4f}, F1: {aff_f1:.4f}\n")
        f.write(f"test_total_time: {test_total_time:.4f}\n")
        f.write(f"total_train_time: {self.total_train_time:.4f}\n")
        f.write('nmae:{}, nrmse:{}, kl:{}'.format(nmae, nrmse, kl))
        f.write('\n')
        f.write('Total:{}, Trainable:{} '.format(self.total_params, self.trainable_params))
        f.write('\n\n')
        f.close()

        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('nmae:{}, nrmse:{} kl:{}'.format(nmae, nrmse, kl))
        f.write('\n\n')
        f.close()

        return