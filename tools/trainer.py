import os
import torch
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
        self.model.cuda()
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
        self.class_correct = log_cfg.get('class_correct', False)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = open(os.path.join(self.log_dir, log_cfg['log_file']), 'w')

        self.accumulate_batch_size = train_cfg.get('accumulate_batch_size', -1)
        self.with_accumulate_batch = self.accumulate_batch_size > 0
        self.accu_counter = 0
        if self.with_accumulate_batch:
            assert self.accumulate_batch_size > self.train_dataloader.batch_size
            assert self.accumulate_batch_size % self.train_dataloader.batch_size == 0
            self._log('Using accumulate batch. Small batch size: {}. Accumulate batch size: {}'.format(
                self.train_dataloader.batch_size, self.accumulate_batch_size))
            self.update_freq = self.accumulate_batch_size // self.train_dataloader.batch_size
        
        load_from = train_cfg.get('load_from', None)
        if load_from is not None:
            assert os.path.exists(load_from)
            state = torch.load(load_from)
            model.load_state_dict(state['model_params'])
            self._log('load state dict: {}'.format(load_from))
            
        checkpoint = train_cfg['checkpoint']
        if checkpoint is not None:
            assert os.path.exists(checkpoint)
            state = torch.load(checkpoint)
            model.load_state_dict(state['model_params'])
            self.start_epoch = state['epoch'] + 1
            self.best_score = state.get('best_score', state['score'])
            self._log('load checkpoint: {}.\nepoch: {}    score: {}'.format(
                checkpoint, self.start_epoch, self.best_score))
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
        for epoch in range(self.start_epoch, self.epoch + 1):
            self.lr_scheduler.epoch_schedule(epoch)
            self._log('epoch: {} | lr: {}'.format(epoch, self.lr_scheduler.base_lr[0]))
            self._train_one_epoch(epoch)
            if epoch % self.val_frequency == 0:
                score = self._validate()
                self._log('epoch: {} | validate score: {:.6f}'.format(epoch, score))
                if self.best_score < score:
                    self.best_score = score
                    self.best_epoch = epoch
                    self._log('best_epoch: {} | best_score: {}'.format(self.best_epoch, self.best_score))
                    self._save_checkpoint(epoch, score, name='best_model')
            if epoch % self.save_frequency == 0:
                self._save_checkpoint(epoch, score, name='epoch_{}'.format(epoch))
            else:
                self._save_checkpoint(epoch, score, name='latest_model')

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

            pred = self.model(data)
            loss = self.criterion(pred, label)
            loss_val = loss.item()

            if self.print_frequency != 0 and is_log:
                self._log('epoch: {} | iter: {} | loss: {:.6f}'.format(
                    epoch, self.cur_iter, loss_val))

            self._update_params(loss)


    def _validate(self):
        self.model.eval()
        total_sample, total_correct = 0, 0
        if self.class_correct:
            correct_dict = {k: 0 for k in range(1108)}

        with torch.no_grad():
            for data, label in self.val_dataloader:
                data = data.cuda()
                label = label.cuda()

                output = self.model.forward_test(data)
                pred = output.argmax(dim=1)
                correct = pred == label

                total_sample += label.size(0)
                total_correct += correct.sum().item()

                if self.class_correct:
                    correct_label = label[correct].cpu().numpy()
                    for cl in correct_label:
                        num = correct_dict.get(cl, 0)
                        correct_dict[cl] = num + 1

        if self.class_correct:
            for k, v in correct_dict.items():
                self._log('class{} : {}/{}'.format(k, v, self.val_dataloader.dataset.num_dict[k]))

        return total_correct / total_sample


    def _save_checkpoint(self, epoch, score, name='checkpoint'):
        state = {
            'iter': self.cur_iter,
            'epoch': epoch,
            'best_score': self.best_score,
            'score': score,
            'model_params': self.model.state_dict()
        }
        torch.save(state, os.path.join(self.log_dir, '{}.pth'.format(name)))