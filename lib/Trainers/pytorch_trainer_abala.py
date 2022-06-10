import numpy as np
import torch
import pathlib

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 metric: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 save_chkpoint: bool = False,
                 model_name: str ='SNET'
                 ):

        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.save_chkpoint = save_chkpoint
        self.model_name = model_name
        self.training_loss = []
        self.validation_loss = []
        self.training_acc = []
        self.validation_acc = []
        self.learning_rate = []

    def run_trainer(self):

        from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.step(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.step()  # learning rate scheduler step
        return self.training_loss, self.training_acc, self.validation_loss,self.validation_acc, self.learning_rate

    def _train(self):
        from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []   # accumulate the losses here
        train_acc = []      # accumulate the accuracy here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            acc = self.metric(out, target)  # calculate accuracy
            acc_value = acc.item()
            train_acc.append(acc_value)
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
            batch_iter.set_description(f'Training: (acc {acc_value:.4f})')  # update progressbar
            
        batch_iter.set_description(f'Epoch Training loss: {np.mean(train_losses):.4f}')
        batch_iter.set_description(f'Epoch Training accuracy: {np.mean(train_acc):.4f}')  # update progressbar
        self.training_loss.append(np.mean(train_losses))
        self.training_acc.append(np.mean(train_acc))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_acc = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                acc = self.metric(out, target)  # calculate accuracy
                acc_value = acc.item()
                valid_acc.append(acc_value)
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
                batch_iter.set_description(f'Validation: (acc {acc_value:.4f})')  # update progressbar
        batch_iter.set_description(f'Epoch validation loss: {np.mean(valid_losses):.4f}')
        batch_iter.set_description(f'Epoch validation accuracy: {np.mean(valid_acc):.4f}')  # update progressbar

        self.validation_loss.append(np.mean(valid_losses))
        self.validation_acc.append(np.mean(valid_acc))
        # model_name = '{}_{}.pt'.format(self.model_name, np.round(np.mean(valid_acc),3))
        # torch.save(self.model.state_dict(), pathlib.Path.cwd() / model_name)

        batch_iter.close()