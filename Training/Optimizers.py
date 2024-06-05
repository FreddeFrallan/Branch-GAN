from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import optim
import torch

NAME_TO_OPTIM = {'adam': optim.Adam,
                 'adagrad': optim.Adagrad,
                 'sgd': optim.SGD,
                 'rmsprop': optim.RMSprop}


def createTorchOptimizer(modelParams, name, lr, zeroOptimizer=False, **params):
    if (zeroOptimizer):
        print("Creating zero optimizer")
        return ZeroRedundancyOptimizer(modelParams, optimizer_class=NAME_TO_OPTIM[name.lower()], lr=lr, **params)
    return NAME_TO_OPTIM[name.lower()](modelParams, lr=lr, **params)


def _createScheduler(schedulerType, schedulerArgs, optimizer, gradAccum, trainDataLoader, epochs):
    if (schedulerType is None or schedulerArgs is None):
        print("No lr scheduler specified")
        return None
    print("Creating lr scheduler!")
    totalSteps = int(len(trainDataLoader) * epochs / gradAccum)
    stepSizeUp = schedulerArgs.get('step_size_up', totalSteps)
    stepSizeDown = totalSteps - stepSizeUp
    return schedulerType(optimizer, **schedulerArgs, step_size_down=stepSizeDown)


def createOptimizationWrappers(model, optimConfigs, trainDataLoader, epochs, mixedPrecision=False):
    torchOptims = []
    for conf in optimConfigs:
        torchOptims.append(
            createTorchOptimizer(model.parameters(), conf.name, conf.lr, conf.useZeroOptimizer, **conf.params))

    if (len(torchOptims) == 1):
        conf = optimConfigs[0]
        print(conf.schedulerArgs)
        scheduler = _createScheduler(conf.schedulerType, conf.schedulerArgs, torchOptims[0], conf.gradAccum,
                                     trainDataLoader, epochs)
        return OptimizerWrapper(model, torchOptims[0], conf.gradAccum, conf.lossKey, scheduler,
                                mixedPrecision=mixedPrecision)
    else:
        firstConf = optimConfigs[0]
        print(firstConf.schedulerArgs)
        key2optim = {conf.lossKey: optim for conf, optim in zip(optimConfigs, torchOptims)}
        return MultipleOptimizerWrapper(key2optim, firstConf.gradAccum)


class OptimizerWrapper:

    def __init__(self, model, optimizer, gradAccum, lossKey, scheduler=None, mixedPrecision=False):
        self.model = model
        self.optimizer = optimizer
        self.gradAccum = gradAccum
        self.lossKey = lossKey
        self.scheduler = scheduler
        self.mixedPrecision = mixedPrecision
        if (mixedPrecision):
            print("Creating mixed precision optimizer")
            self.scaler = torch.cuda.amp.GradScaler()

    def clearGrad(self):
        print("Claering grad")
        self.optimizer.zero_grad()

    def performUpdateStep(self, losses, stepCounter):
        if (self.mixedPrecision):
            return self._mixedPrecisionStep(losses, stepCounter)
        loss = losses[self.lossKey]
        loss.backward()

        if (stepCounter % self.gradAccum == 0):
            self.optimizer.step()
            self.optimizer.zero_grad()
            if (self.scheduler is not None):
                self.scheduler.step()

    def _mixedPrecisionStep(self, losses, stepCounter):
        loss = losses[self.lossKey]
        self.scaler.scale(loss).backward()

        if (stepCounter % self.gradAccum == 0):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if (self.scheduler is not None):
                self.scheduler.step()

    def saveOptimizer(self, savePath):
        torch.save(self.optimizer.state_dict(), savePath)


class MultipleOptimizerWrapper:

    def __init__(self, optimizersAndKeys, gradAccum):
        self.optimizersAndKeys = optimizersAndKeys
        self.numOptimizers = len(self.optimizersAndKeys)
        self.gradAccum = gradAccum

    def clearGrad(self):
        for optimizer in self.optimizersAndKeys.values():
            optimizer.zero_grad()

    def performUpdateStep(self, losses, stepCounter):
        for i, (key, optimizer) in enumerate(self.optimizersAndKeys.items()):
            # print(i, key)
            loss = losses[key]
            # if (i + 1 == self.numOptimizers):
            # loss.backward()
            # else:
            loss.backward(retain_graph=True)
            break

        if (stepCounter % self.gradAccum == 0):
            for optimizer in self.optimizersAndKeys.values():
                optimizer.step()
                break
            for optimizer in self.optimizersAndKeys.values():
                optimizer.zero_grad()
                break


class FreezeLRSchedule(_LRScheduler):
    def __init__(self, optimizer, zero_steps, warmup_steps, lr_start, lr_end, last_epoch=-1, **kwargs):
        self.zero_steps = zero_steps
        self.warmup_steps = warmup_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super(FreezeLRSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.zero_steps:
            return [0 for base_lr in self.base_lrs]

        elif self.last_epoch < self.zero_steps + self.warmup_steps:
            warmup_fraction = (self.last_epoch - self.zero_steps) / self.warmup_steps
            return [self.lr_start + warmup_fraction * (self.lr_end - self.lr_start) for base_lr in self.base_lrs]

        else:
            return [self.lr_end for base_lr in self.base_lrs]