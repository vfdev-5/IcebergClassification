
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler


class CosineWithRestartLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by a cosine function with restart option (as in SGDR paper). When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        restart_every (int): Restart cosine function every `restart_every` epoch.
            By default, it can be chosen to equal number of epochs.
        restart_factor (float): factor to rescale `e_restart` after each restart.
        min_lr (float): minimum learning rate
        last_epoch (int): The index of last epoch. Default: -1.
        
    Learning rate decay formula:
    ```
    t[-1] = 0
    ...
    t[e] = t[e-1] + 1 
    if t[e] % restart_every == 0:
        t[e] = 0
        restart_every *= restart_factor
    lr[e] = (base_lr - min_lr) * ( 1.0 + cos(2.0 * pi / period * t[e]) ) * 0.5 + min_lr
    ```
    """

    def __init__(self, optimizer, restart_every, restart_factor=1.0, min_lr=0.0, last_epoch=-1, verbose=False):        
        self.restart_every = restart_every
        self.restart_factor = restart_factor
        self.min_lr = min_lr
        self._t = -1
        self.verbose = verbose
        super(CosineWithRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self._t
        self._t += 1
        if self.restart_every > 0 and self.last_epoch > 0 and \
            self._t % self.restart_every == 0:
            self._t = 0
            self.restart_every = int(self.restart_every * self.restart_factor)            
            if self.verbose:
                print("\nCosineWithRestartLR: restart lr at epoch %i, next restart at %i" \
                    % (self.last_epoch, self.last_epoch + self.restart_every))

        return [(base_lr - self.min_lr) * (1.0 + np.cos( np.pi / self.restart_every * t )) * 0.5 + self.min_lr
                for base_lr in self.base_lrs]


class LRSchedulerWithRestart(_LRScheduler):
    """Proxy learning scheduler with restarts: learning rate follows input scheduler strategy but
    the strategy can restart when passed a defined number of epochs. Ideas are taken from SGDR paper.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        scheduler (_LRScheduler): input lr scheduler
        restart_every (int): Restart cosine function every `restart_every` epoch.
            By default, it can be chosen to equal number of epochs.
        restart_factor (float): factor to rescale `e_restart` after each restart.        
        last_epoch (int): The index of last epoch. Default: -1.
        
    Learning rate strategy formula:
    ```
    t[-1] = 0
    ...
    t[e] = t[e-1] + 1 
    if t[e] % restart_every == 0:
        t[e] = 0
        restart_every *= restart_factor
    scheduler.last_epoch = t[e]
    lr[e] = scheduler.get_lr()
    ```
    """

    def __init__(self, scheduler, restart_every, restart_factor=1.0, last_epoch=-1, verbose=False):        
        self.scheduler = scheduler
        self.restart_every = restart_every
        self.restart_factor = restart_factor        
        self._t = -1
        self.verbose = verbose
        # Do not call super method as optimizer is already setup by input scheduler
        # super(LRSchedulerWithRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self._t
        self._t += 1
        if self.restart_every > 0 and self.last_epoch > 0 and \
            self._t % self.restart_every == 0:
            self._t = 0
            self.restart_every = int(self.restart_every * self.restart_factor)            
            if self.verbose:
                print("\LRSchedulerWithRestart: restart lr at epoch %i, next restart at %i" \
                    % (self.last_epoch, self.last_epoch + self.restart_every))
        
        self.scheduler.last_epoch = self._t
        return self.scheduler.get_lr()
    
    def step(self, epoch=None):
        self.scheduler.step(epoch)
