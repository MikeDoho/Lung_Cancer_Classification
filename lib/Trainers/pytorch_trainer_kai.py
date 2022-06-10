# Python Modules
import numpy as np
import time
import math
from sklearn.metrics import roc_auc_score


# Torch Modules
import torch
import torch.nn.functional as F

# Self Modules
from lib.utils.general import prepare_input
from lib.utils.logger import log


# from lib.visual3D_temp.BaseWriter import TensorboardWriter

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


# def accuracy(output, target):
#     """Computes the accuracy for multiple binary predictions"""
#     pred = output.item() >= 0.5
#     truth = target >= 0.5
#     acc = pred.eq(truth).sum() / target.numel()
#     return acc

def auc_cal(output, target):
    with torch.no_grad():
        auc = roc_auc_score(y_true=target, y_score=output)
    return auc


def calculate_auc(outputs, targets):
    with torch.no_grad():
        outputs = torch.sigmoid(outputs)
        # auc = calculate_auc(outputs.type(torch.FloatTensor).cpu().data.numpy(), targets.cpu().numpy())
        try:
            auc = roc_auc_score(targets.cpu().numpy(), outputs.type(torch.FloatTensor).cpu().data.numpy()[:, 1])
            return auc
        except:
            return np.nan


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion_pre, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion_pre = criterion_pre
        # self.metric = metric
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(train_data_loader.batch_size))
        # self.writer = TensorboardWriter(args)

        self.save_frequency = 20
        # self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1
        self.val_loss = 0

        self.print_batch_spacing = 10
        self.acculumation_steps = 5

    def training(self):
        # self.model.train()


        for epoch in range(self.start_epoch, self.args.nEpochs):

            for param_group in self.optimizer.param_groups:
                lr_show = param_group['lr']
            print('\n########################################################################')
            print(f"Training epoch: {epoch}, Learning rate: {lr_show:.8f}")

            self.train_epoch_alex(epoch)

            if self.do_validation:
                print(f"Validation epoch: {epoch}")
                self.validate_epoch_alex(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # # comment out for speed test
            # if epoch % self.save_frequency == 0:
            #     self.model.save_checkpoint(self.args.save,
            #                            epoch, self.val_loss,
            #                            optimizer=self.optimizer)
            print('\n\n')

    def train_epoch_alex(self, epoch):

        # Creates once at the beginning of training
        # scaler = torch.cuda.amp.GradScaler()

        def time_report(initial_time, time_name):
            get_time_diff = time.gmtime(time.time() - initial_time)
            readable_time = time.strftime("%M:%S", get_time_diff)
            print(f"{time_name} time: {readable_time} (min:seconds)")
            del get_time_diff
            del readable_time

        epoch_start_time = time.time()
        self.model.train()
        time_report(epoch_start_time, 'model.train()')

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        acc_cum = []
        auc_cum = []

        print('-------------------------------------------------------------------------------------------')

        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            batch_timer = time.time()
            time_report(epoch_start_time, 'enter batch loop')


            # Gathering input data; prepare_input sends to gpu
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            time_report(batch_timer, 'input manipulation')
            input_tensor.requires_grad = True
            time_report(batch_timer, 'grad equals True')
            # Model make prediction
            pred = self.model(input_tensor)
            time_report(batch_timer, 'input and model prediction')

            # calculating loss and metrics
            # with torch.cuda.amp.autocast():
            loss = self.criterion_pre(pred, target.long().view(-1))/self.acculumation_steps
            time_report(batch_timer, 'loss calculation')

            # need to calculate gradient
            self.model.zero_grad()

            loss.backward()
            # scaler.scale(loss).backward()
            time_report(batch_timer, 'gradient calculation')

            # (loss_growth*self.growth_weight + loss_dice).backward()
            def clip_gradient(optimizer, grad_clip):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)


            clip_gradient(self.optimizer, 5)
            self.optimizer.step()
            # scaler.step(self.optimizer)
            # scaler.update()

            time_report(batch_timer, 'grad clip and optim step')

            # with torch.no_grad():

            #Calculating and appending
            with torch.no_grad():
                acc = calculate_accuracy(pred, target.long().view(-1))
                time_report(batch_timer, 'acc')
                # acc = 0
                auc = calculate_auc(pred, target)
                time_report(batch_timer, 'auc')

                # storing loss and metrics
                loss_cum.append(loss.item())
                time_report(batch_timer, 'store loss')

                acc_cum.append(acc)
                auc_cum.append(auc)
                time_report(batch_timer, 'store acc and auc')

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    print('\t**************************************************************************')
                    print(f"\tBatch {batch_idx + 1} of {len(self.train_data_loader)}")
                    print(f"\tLoss: {loss.item()}, ACC: {acc}, AUC: {auc}")
                    print('\t**************************************************************************')
                else:
                    pass

            time_report(batch_timer, 'finish one batch')
            print('\n\n')

        # Calculating time per epoch
        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        print(f"Training epoch completed in {res} (min:seconds)")
        # self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)
        print(f"Summary-----Loss: {sum(loss_cum) / len(loss_cum)}, ACC: {sum(acc_cum) / len(acc_cum)}, AUC: {np.nanmean(auc_cum)}")
        print(f"Number of np.nan in AUC list: {sum(math.isnan(x) for x in auc_cum)}")
        print('-------------------------------------------------------------------------------------------')

    def validate_epoch_alex(self, epoch):
        self.model.eval()

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        acc_cum = []
        auc_cum = []

        # starting epoch timer
        epoch_start_time = time.time()

        print('-------------------------------------------------------------------------------------------')
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):

            if (batch_idx + 1) % self.print_batch_spacing == 0:
                print('*************************************')
                print(f"\tBatch {batch_idx + 1} of {len(self.valid_data_loader)}")
            else:
                pass

            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False

                pred = self.model(input_tensor)

                # calculating loss and metrics
                loss = self.criterion_pre(pred, target.long().view(-1))
                acc = calculate_accuracy(pred, target.long().view(-1))
                auc = calculate_auc(pred, target)
                # auc = auc_cal(output=pred[:, 1].detach().cpu().numpy(),
                #              target=target.long().view(-1).detach().cpu().numpy())

                # print("\tLoss: {:.4f}, ACC: {:.4f}".format(loss.item(), acc))
                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    print(f"\tLoss: {loss.item()}, ACC: {acc}, AUC: {auc}")
                    print('*************************************')
                else:
                    pass

                # storing loss and metrics
                loss_cum.append(loss.item())
                acc_cum.append(acc)
                auc_cum.append(auc)

        self.val_loss = sum(loss_cum) / len(loss_cum)

        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        print(f"Validation epoch completed in {res} (min:seconds)")
        print(f"Summary-----Loss: {sum(loss_cum) / len(loss_cum)}, ACC: {sum(acc_cum) / len(acc_cum)}, AUC: {np.nanmean(auc_cum)}")
        print(f"Number of np.nan in AUC list: {sum(math.isnan(x) for x in auc_cum)}")
        print('-------------------------------------------------------------------------------------------')
