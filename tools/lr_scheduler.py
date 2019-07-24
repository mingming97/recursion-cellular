class LrScheduler(object):

    def __init__(self, optimizer, lr_cfg):
        self.warmup = lr_cfg.get('warmup', None)
        self.warmup_iters = lr_cfg.get('warmup_iters', 0)
        self.warmup_ratio = lr_cfg.get('warmup_ratio', 0.1)
        if self.warmup is not None:
            assert self.warmup in ['constant', 'linear', 'exp']
            assert self.warmup_iters > 0
            assert 0 < self.warmup_ratio <= 1.0
        self.gamma = lr_cfg.get('gamma', 0.1)
        self.step = lr_cfg.get('step', [])
        self.optimizer = optimizer
        self.base_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        self.warmup_lr = []

    def _get_warmup_lr(self, cur_iter):
        if self.warmup == 'constant':
            self.warmup_lr = [_lr * self.warmup_ratio for _lr in self.base_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iter / self.warmup_iters) * (1 - self.warmup_ratio)
            self.warmup_lr = [_lr * (1 - k) for _lr in self.base_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iter / self.warmup_iters)
            self.warmup_lr = [_lr * k for _lr in self.base_lr]

    def _get_epoch_lr(self, cur_epoch):
        if cur_epoch in self.step:
            self.base_lr = [_lr * self.gamma for _lr in self.base_lr]

    def _set_lr(self, lr):
        for param_group, _lr in  zip(self.optimizer.param_groups, lr):
            param_group['lr'] = _lr

    def iter_schedule(self, cur_iter, is_print=False):
        if self.warmup is not None and cur_iter < self.warmup_iters:
            self._get_warmup_lr(cur_iter)
            if is_print:
                print('warmup iter: {} | lr: {}'.format(cur_iter, self.warmup_lr[0]))
            self._set_lr(self.warmup_lr)

    def epoch_schedule(self, cur_epoch):
        self._get_epoch_lr(cur_epoch)
        self._set_lr(self.base_lr)