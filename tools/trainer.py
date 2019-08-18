import os
import torch
import numpy as np
from tqdm import tqdm
from .lr_scheduler import LrScheduler


class Trainer:
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 val_dataloader, 
                 criterion, 
                 optimizer,
                 train_cfg, 
                 log_cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.epoch = train_cfg['epoch']
        self.log_dir = log_cfg['log_dir']
        self.print_frequency = log_cfg['print_frequency']
        self.val_frequency = log_cfg.get('val_frequency', 1)
        self.save_frequency = log_cfg.get('save_frequency', 10)
        self.lr_cfg = train_cfg['lr_cfg']
        self.mix_up = train_cfg.get('mix_up', False)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = open(os.path.join(self.log_dir, log_cfg['log_file']), 'w')

        if self.mix_up:
            self._log('Using mix up')

        self.accumulate_batch_size = train_cfg.get('accumulate_batch_size', -1)
        self.with_accumulate_batch = self.accumulate_batch_size > 0
        self.accu_counter = 0
        if self.with_accumulate_batch:
            assert self.accumulate_batch_size > self.train_dataloader.batch_size
            assert self.accumulate_batch_size % self.train_dataloader.batch_size == 0
            self._log('Using accumulate batch. Small batch size: {}. Accumulate batch size: {}'.format(
                self.train_dataloader.batch_size, self.accumulate_batch_size))
            self.update_freq = self.accumulate_batch_size // self.train_dataloader.batch_size
             
        checkpoint = train_cfg['checkpoint']
        if checkpoint is not None:
            assert os.path.exists(checkpoint)
            state = torch.load(checkpoint)
            model.load_state_dict(state['model_params'], strict=False)
            self.start_epoch = state['epoch'] + 1
            self.best_score = state['score']
            self._log('load checkpoint.\nepoch: {}    score: {}'.format(
                self.start_epoch, self.best_score))
            self.cur_iter = state.get('iter', 1)
            self.lr_cfg['warmup'] = None
            self.best_epoch = state.get('best_epoch', -1)
        else:
            self.start_epoch = 1
            self.best_score = 0
            self.best_epoch = -1
            self.cur_iter = 1

        self.lr_scheduler = LrScheduler(self.optimizer, self.lr_cfg)


    def _log(self, logstr):
        print(logstr)
        self.log_file.write(str(logstr))
        self.log_file.write('\n')
        self.log_file.flush()

    def train(self):
        for epoch in range(self.start_epoch, self.epoch):
            self.lr_scheduler.epoch_schedule(epoch)
            self._log('epoch: {} | lr: {}'.format(epoch, self.lr_scheduler.base_lr[0]))
            self._train_one_epoch(epoch)
            if self.epoch % self.val_frequency == 0:
                score = self._validate()
                self._log('epoch: {} | validate score: {:.6f}'.format(epoch, score))
                if self.best_score < score:
                    self.best_score = score
                    self.best_epoch = epoch
                    self._log('best_epoch: {} | best_score: {}'.format(self.best_epoch, self.best_score))
                    self._save_checkpoint(epoch, score, name='best_model')
            if epoch % self.save_frequency == 0:
                self._save_checkpoint(epoch, score, name='epoch_{}'.format(epoch))


    def _update_params(self, loss):
        if not self.with_accumulate_batch:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.cur_iter += 1
        else:
            self.accu_counter += 1
            loss = loss / self.update_freq
            loss.backward()
            if self.accu_counter == self.update_freq:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accu_counter = 0
                self.cur_iter += 1


    def _train_one_epoch(self, epoch):
        self.model.train()
        for i, (data, label) in enumerate(self.train_dataloader):
            is_log = self.cur_iter % self.print_frequency == 0 and self.accu_counter == 0
            self.lr_scheduler.iter_schedule(self.cur_iter, is_log)
            data = data.cuda()
            label = label.cuda()

            if self.mix_up:
                lambda_ = np.random.beta(0.2, 0.2)
                new_perm = torch.randperm(data.size(0))
                mix_data = lambda_ * data + (1 - lambda_) * data[new_perm]
                new_label = label[new_perm]
                pred = self.model(mix_data)
                loss = lambda_ * self.criterion(pred, label) + (1 - lambda_) * self.criterion(pred, new_label)
            else:
                pred = self.model(data, label)
                loss = self.criterion(pred, label)

            loss_value = loss.item()

            if self.print_frequency != 0 and is_log:
                self._log('epoch: {} | iter: {} | loss: {:.6f}'.format(epoch, self.cur_iter, loss_value))

            self._update_params(loss)


    def _validate(self):
        self.model.eval()
        total_sample, total_correct = 0, 0
        correct_dict = {k: 0 for k in range(1108)}

        # compute center features
        center_feat = None if self.model.extra_module is None else [[] for i in range(1108)]
        if center_feat is not None:
            with torch.no_grad():
                for data, label in self.train_dataloader:
                    data = data.cuda()
                    feat = self.model.forward_test(data).cpu().numpy()
                    for l, f in zip(label, feat):
                        center_feat[int(l.item())].append(f)
            for i in range(1108):
                center_feat[i] = np.mean(np.array(center_feat[i]), axis=0)
            center_feat = np.array(center_feat)

        with torch.no_grad():
            for data, label in self.val_dataloader:
                data = data.cuda()
                label = label.cuda()

                output = self.model.forward_test(data, center_feat).cuda()
                pred = output.argmax(dim=1)
                correct = pred == label

                total_sample += label.size(0)
                total_correct += correct.sum().item()

                # correct_label = label[correct].cpu().numpy()
                # for cl in correct_label:
                #     num = correct_dict.get(cl, 0)
                #     correct_dict[cl] = num + 1

        # for k, v in correct_dict.items():
        #     self._log('class{} : {}/{}'.format(k, v, self.val_dataloader.dataset.num_dict[k]))

        return total_correct / total_sample


    def _save_checkpoint(self, epoch, score, name='checkpoint'):
        state = {
            'iter': self.cur_iter,
            'epoch': epoch,
            'score': score,
            'model_params': self.model.state_dict()
        }
        torch.save(state, os.path.join(self.log_dir, '{}.pth'.format(name)))