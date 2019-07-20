import os
import torch
import numpy as np


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
        self.lr_cfg = train_cfg['lr_cfg']
        self.mix_up = train_cfg.get('mix_up', False)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = open(os.path.join(self.log_dir, log_cfg['log_file']), 'w')

        if self.mix_up:
            self._log('Using mix up')

        self.accumulate_batch_size = train_cfg.get('accumulate_batch_size', -1)
        self.with_accumulate_batch = self.accumulate_batch_size > 0
        if self.with_accumulate_batch:
            assert self.accumulate_batch_size > self.train_dataloader.batch_size
            assert self.accumulate_batch_size % self.train_dataloader.batch_size == 0
            self._log('Using accumulate batch. Small batch size: {}. Accumulate batch size: {}'.format(
                self.train_dataloader.batch_size, self.accumulate_batch_size))
            self.update_freq = self.accumulate_batch_size // self.train_dataloader.batch_size
            self.accu_counter = 0
             
        checkpoint = train_cfg['checkpoint']
        if checkpoint is not None:
            assert os.path.exists(checkpoint)
            state = torch.load(checkpoint)
            model.load_state_dict(state['model_params'])
            self.start_epoch = state['epoch'] + 1
            self.best_score = state['score']
            self._log('load checkpoint.\nepoch: {}    score: {}'.format(
                self.start_epoch, self.best_score))
        else:
            self.start_epoch = 0
            self.best_score = 0

    def _log(self, logstr):
        print(logstr)
        self.log_file.write(str(logstr))
        self.log_file.write('\n')
        self.log_file.flush()


    def _lr_schedule(self, epoch):
        base_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        lr_step = self.lr_cfg['step']
        if epoch in lr_step:
            for param_group, lr in zip(self.optimizer.param_groups, base_lr):
                param_group['lr'] = lr * self.lr_cfg['gamma']


    def train(self):
        for epoch in range(self.start_epoch, self.epoch):
            self._lr_schedule(epoch)
            self._train_one_epoch(epoch)
            score = self._validate()
            self._log('epoch: {} | validate score: {:.6f}'.format(epoch, score))
            if self.best_score < score:
                self.best_score = score
                self._save_checkpoint(epoch, score)


    def _update_params(self, loss):
        if not self.with_accumulate_batch:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.accu_counter += 1
            loss = loss / self.update_freq
            loss.backward()
            if self.accu_counter == self.update_freq:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accu_counter = 0


    def _train_one_epoch(self, epoch):
        self.model.train()
        for i, (data, label) in enumerate(self.train_dataloader):
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
                pred = self.model(data)
                loss = self.criterion(pred, label)

            loss_value = loss.item()

            self._update_params(loss)

            if self.print_frequency != 0 and (i + 1) % self.print_frequency == 0:
                self._log('epoch: {} | small iter: {} | loss: {:.6f}'.format(epoch, i + 1, loss_value))


    def _validate(self):
        self.model.eval()
        total_sample, total_correct = 0, 0
        correct_dict = {k: 0 for k in range(1108)}
        with torch.no_grad():
            for data, label in self.val_dataloader:
                data = data.cuda()
                label = label.cuda()

                output = self.model(data)
                pred = output.argmax(dim=1)
                correct = pred == label

                total_sample += label.size(0)
                total_correct += correct.sum().item()

                correct_label = label[correct].cpu().numpy()
                for cl in correct_label:
                    num = correct_dict.get(cl, 0)
                    correct_dict[cl] = num + 1

        for k, v in correct_dict.items():
            self._log('class{} : {}/{}'.format(k, v, self.val_dataloader.dataset.num_dict[k]))

        return total_correct / total_sample

    def _save_checkpoint(self, epoch, score):
        state = {
            'epoch': epoch,
            'score': score,
            'model_params': self.model.state_dict()
        }
        torch.save(state, os.path.join(self.log_dir, 'checkpoint.pth'))
