import logging
from datetime import datetime

import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import device, amp
from typing import List, Dict
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data import DataLoader
from pyseizure.classifier.classifier import Classifier
from pyseizure.data_classes.classifier import Classifier as classifier_name
from pyseizure.helpers.conv_dataset import ConvDataset
from pyseizure.helpers.metrics import get_confusion_matrix, get_mse, get_roc, \
    get_accuracy, get_precision, get_f1, get_prauc, get_sensitivity, \
    get_specificity, PostProcessor
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.helpers.supervised_contrastive_loss import \
    SupervisedContrastiveLoss
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector
from sklearn.metrics import auc

VERBOSITY = 0


class EEGNet(Classifier):
    def __init__(self, train_samples: List[str], valid_samples: List[str],
                 test_samples: List[str], normaliser: Normaliser = None,
                 feature_selector: FeatureSelector = None,
                 eval_only: bool = False, config: Dict = None):
        super().__init__(train_samples, test_samples, valid_samples,
                         normaliser, feature_selector, eval_only, config)
        self.config = {
            'conv1_hd': 64,
            'conv1_k': 3,
            'conv2_k': 3,
            'avg_pool_x': 4,
            'avg_pool_y': 1,
            'f1_hd': 256,
            'weight_decay': 1e-3,
            'lr': 1e-4,
            'batch_size': 32,
            'frequency': 256,
            'channels': self.no_channels,
            'optimiser': 'Adam'
        }

        self._init_loaders_and_model()

    def _init_loaders_and_model(self):
        eval_only = False if self.eval_only else True

        self.dataset = {
            'train': ConvDataset(self.train_samples,
                                 self.label_encoder,
                                 self.feature_selector,
                                 self.normaliser,
                                 batch_size=self.config['batch_size'],
                                 channels=self.config['channels'],
                                 use_scoring=self.use_scoring,
                                 subsampling=eval_only,
                                 binary=self.binary,
                                 artefacts=self.artefacts),
            'eval': ConvDataset(self.valid_samples,
                                self.label_encoder,
                                self.feature_selector,
                                self.normaliser,
                                batch_size=self.config['batch_size'],
                                channels=self.config['channels'],
                                use_scoring=self.use_scoring,
                                subsampling=eval_only,
                                binary=self.binary,
                                artefacts=self.artefacts),
            'test': ConvDataset(self.test_samples,
                                self.label_encoder,
                                self.feature_selector,
                                self.normaliser,
                                batch_size=self.config['batch_size'],
                                channels=self.config['channels'],
                                use_scoring=False,
                                subsampling=False,
                                binary=self.binary,
                                artefacts=self.artefacts),
        }
        self.data_loader = {
            'train': DataLoader(self.dataset['train'],
                                batch_size=self.config['batch_size'],
                                shuffle=eval_only,
                                num_workers=self.no_threads_per_core,
                                pin_memory=True,
                                drop_last=True),
            'valid': DataLoader(self.dataset['eval'],
                                batch_size=self.config['batch_size'],
                                shuffle=eval_only,
                                num_workers=self.no_threads_per_core,
                                pin_memory=True,
                                drop_last=True),
            'test': DataLoader(self.dataset['test'],
                               batch_size=self.config['batch_size'],
                               shuffle=False,
                               num_workers=self.no_threads_per_core,
                               pin_memory=True),
        }
        self.model = EEGNetModel(self.dev, self.config).to(self.dev)

        self.optimiser = getattr(torch.optim, self.config['optimiser'])(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'])
        if self.use_scheduler:
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimiser,
                T_max=5 * len(self.data_loader['train']),
                eta_min=1e-6)
        else:
            self.scheduler = None
        if self.use_contrastive:
            self.criterion = SupervisedContrastiveLoss(temperature=.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.dev)

    def __del__(self):
        del self.dataset

    def load_model(self, model_path: str = ''):
        self.model = torch.load(model_path, weights_only=False)

    def tune(self, config: Dict):
        logging.info('Model not tunable')

    @staticmethod
    def plot_single_epoch(inputs:torch.Tensor, labels: torch.Tensor):
        for i in range(inputs.shape[1]):
            plt.figure(figsize=(10, 5))
            plt.imshow(inputs[0][i].numpy(), interpolation='nearest',
                       aspect='auto')
            plt.title(f"Epoch: {i}, class: {labels[0][i][0][0].numpy()}")
            plt.tight_layout()
            plt.show()

    def train(self):
        writer = SummaryWriter(
            f"output/runs/{classifier_name.EEGNET.value.lower()}"
            f"_{self.dataset_name}_"
            f"{datetime.now().strftime('%y-%m-%d_%H_%M_%S')}")
        auroc = torchmetrics.AUROC(task='multiclass',
                                   num_classes=2).to(self.dev)
        roc = torchmetrics.ROC(task='multiclass',
                               num_classes=2).to(self.dev)
        f1 = torchmetrics.AveragePrecision(task="multiclass",
                                           num_classes=2).to(self.dev)
        accuracy = torchmetrics.Accuracy(task="multiclass",
                                         num_classes=2).to(self.dev)
        specificity = torchmetrics.Specificity(task="multiclass",
                                               average='macro',
                                               num_classes=2).to(self.dev)

        self.model.train()
        torch.cuda.empty_cache()
        patience = 40
        best_val_loss, best_val_epoch, best_acc = 1, 0, np.inf
        best_eval_loss, eval_loss = 100, 0
        softmax = nn.Softmax(dim=1)

        with tqdm(total=len(self.data_loader['train']) * self.epoch,
                  unit="batch",
                  bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            for epoch in range(self.epoch):
                pbar.set_description(f"Epoch {epoch:3d}", refresh=True)
                running_loss = 0.0
                f1.reset()
                auroc.reset()
                roc.reset()
                accuracy.reset()
                for i, (inputs, labels) in enumerate(
                        self.data_loader['train']):
                    self.optimiser.zero_grad()
                    with amp.autocast(device_type=self.dev.type, enabled=True):
                        labels = labels.type(torch.LongTensor).to(self.dev)
                        inputs = inputs.squeeze(0).to(self.dev)
                        outputs = self.model(inputs.float())
                        loss = self.criterion(outputs, labels)
                        pred = softmax(outputs)
                        auroc.update(pred, labels)
                        roc.update(pred, labels)
                        f1.update(pred, labels)
                        specificity.update(pred, labels)
                        accuracy.update(pred, labels)

                    if self.scheduler:
                        self.scheduler.step()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                    # self.optimiser.step()

                    running_loss += loss.item()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                    pbar.update(1)

                torch.cuda.empty_cache()
                if writer is not None:
                    writer.add_scalar("Loss/train",
                                      running_loss / ((epoch + 1) * i), epoch)
                    writer.add_scalar("auroc/train", auroc.compute(), epoch)
                    writer.add_scalar("accuracy/train", accuracy.compute(),
                                      epoch)
                    writer.add_scalar("f1/train", f1.compute(), epoch)
                    writer.add_scalar("sensitivity/train",
                                      roc.compute()[1][1].mean().item(), epoch)
                    writer.add_scalar("specificity/train",
                                      specificity.compute(), epoch)
                else:
                    fig, ax = roc.plot(score=True)
                    fig.savefig(f"output/train/"
                                f"{classifier_name.EEGNET.value.lower()}"
                                f"_{epoch}.png")
                    plt.close()
                    pbar.set_postfix(
                        loss=f"{running_loss / ((epoch + 1) * i):.5f}",
                        accuracy=f"{accuracy.compute():.5f}",
                        roc=f"{auroc.compute():.5f}",
                        fpr=f"{roc.compute()[0][1].mean().item():.5f}",
                        sensitivity=f"{roc.compute()[1][1].mean().item():.5f}",
                        specificity=f"{specificity.compute():.5f}",
                        f1=f"{f1.compute():.5f}",
                        refresh=True)

                if epoch % 5 == 0:
                    eval_loss = self.evaluate(self.data_loader['valid'],
                                              epoch, writer)
                    self.model.train()

                if best_eval_loss > eval_loss:
                    best_eval_loss, best_val_epoch = eval_loss, epoch
                    torch.save(self.model,
                               f'output/models/'
                               f'{classifier_name.EEGNET.value.lower()}'
                               f'_{self.dataset_name}.bin')

                if best_val_epoch < (epoch - patience):
                    break

        self.model = torch.load(
            f'output/models/{classifier_name.EEGNET.value.lower()}'
            f'_{self.dataset_name}.bin', weights_only=False)

    def predict(self, data: DataLoader = None,
                return_proba: bool = True) -> np.array:
        """
        Predict output for provided data.

        Parameters
        ----------
        data: torch.utils.data.DataLoader
            input data
        return_proba: bool
            a flag: if true returns probability of each class instead of
            prediction

        Returns
        -------
        numpy.array
            numpy array with predictions and true labels of each class
        """
        self.model.eval()
        if data is None:
            data = self.data_loader['test']

        y_pred_tm = torch.Tensor().to(self.dev)
        y_true_tm = torch.Tensor().to(self.dev)
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for inputs, labels in data:
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.dev)
                inputs = inputs.squeeze(0).to(self.dev)
                outputs = self.model(inputs.float())
                preds = softmax(outputs)
                y_pred_tm = torch.cat((y_pred_tm, preds[:, 1]), 0)
                y_true_tm = torch.cat((y_true_tm, labels), 0)

        return y_pred_tm.cpu().numpy(), y_true_tm.cpu().numpy()

    def evaluate(self,
                 data: DataLoader = None,
                 epoch: int = None,
                 writer: SummaryWriter = None,
                 post_processing: bool = False) -> float:
        self.model.eval()
        if data is None:
            data = self.data_loader['test']
        mse_tm = torchmetrics.MeanSquaredError().to(self.dev)
        auroc_tm = torchmetrics.AUROC(task='multiclass',
                                      num_classes=2).to(self.dev)
        roc_tm = torchmetrics.ROC(task='multiclass',
                                  num_classes=2).to(self.dev)
        precision_tm = torchmetrics.Precision(task='multiclass',
                                              num_classes=2,
                                              average='macro').to(self.dev)
        f1_tm = torchmetrics.AveragePrecision(task='multiclass',
                                              num_classes=2).to(self.dev)
        prc_tm = torchmetrics.PrecisionRecallCurve(task='multiclass',
                                                   num_classes=2).to(self.dev)
        accuracy_tm = torchmetrics.Accuracy(task='multiclass',
                                            num_classes=2).to(self.dev)
        specificity_tm = torchmetrics.Specificity(
            task='multiclass',
            num_classes=2, average='macro').to(self.dev)
        sensitivity_tm = torchmetrics.Recall(task='multiclass',
                                             num_classes=2).to(self.dev)
        confmat = torchmetrics.ConfusionMatrix(task='multiclass',
                                               num_classes=2).to(self.dev)
        running_loss = 0.0
        y_pred_tm = torch.Tensor().to(self.dev)
        y_true_tm = torch.Tensor().to(self.dev)
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for inputs, labels in data:
                labels = labels.type(torch.LongTensor)
                # TODO: use score in post processing
                # scores = labels.squeeze(0)[:, :, 1:]
                labels = labels.to(self.dev)
                inputs = inputs.squeeze(0).to(self.dev)
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                preds = softmax(outputs)
                mse_tm.update(preds[:, 1], labels)
                precision_tm.update(preds, labels)
                auroc_tm.update(preds, labels)
                prc_tm.update(preds, labels)
                roc_tm.update(preds, labels)
                specificity_tm.update(preds, labels)
                sensitivity_tm.update(preds, labels)
                f1_tm.update(preds, labels)
                accuracy_tm.update(preds, labels)

                y_pred_tm = torch.cat((y_pred_tm, preds[:, 1]), 0)
                y_true_tm = torch.cat((y_true_tm, labels), 0)

        y_pred = y_pred_tm.cpu().numpy()
        y_true = y_true_tm.cpu().numpy()
        tn, fp, fn, tp = get_confusion_matrix(y_pred.round(), y_true)
        logging.info(
            f"\nEvaluation (scikit-learn) results:"
            f"\nmse\t\t{get_mse(y_pred, y_true):.5f}\n"
            f"roc\t\t{get_roc(y_pred, y_true):.5f}\n"
            f"accuracy\t{get_accuracy(y_pred, y_true):.5f}\n"
            f"precision\t{get_precision(y_pred, y_true):.5f}\n"
            f"f1\t\t{get_f1(y_pred, y_true):.5f}\n"
            f"prauc\t\t{get_prauc(y_pred, y_true):.5f}\n"
            f"sensitivity\t{get_sensitivity(y_pred, y_true):.5f}\n"
            f"specificity\t{get_specificity(y_pred, y_true):.5f}\n"
            f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")

        if writer is not None:
            writer.add_scalar("auroc/eval", auroc_tm.compute(), epoch)
            writer.add_scalar("accuracy/eval", accuracy_tm.compute(), epoch)
            writer.add_scalar("f1/eval", f1_tm.compute(), epoch)
            writer.add_scalar("prauc/eval", auc(
                prc_tm.compute()[1][1].cpu().numpy(),
                prc_tm.compute()[0][1].cpu().numpy()
            ), epoch)
            writer.add_scalar("mse/eval", mse_tm.compute(), epoch)
            writer.add_scalar("precision/eval", precision_tm.compute(), epoch)
            writer.add_scalar("sensitivity/eval", sensitivity_tm.compute(),
                              epoch)
            writer.add_scalar("specificity/eval", specificity_tm.compute(),
                              epoch)
            writer.add_scalar("Loss/eval",
                              running_loss / len(data), epoch)
        else:
            prauc = auc(prc_tm.compute()[1][1].cpu().numpy(),
                        prc_tm.compute()[0][1].cpu().numpy())
            cm = confmat(y_pred_tm.round(), y_true_tm)
            tn, fp, fn, tp = cm.cpu().numpy().ravel()
            logging.info(f"\nEvaluation (torch_metrics) results:"
                         f"\nmse\t\t{mse_tm.compute():.5f}\n"
                         f"roc\t\t{auroc_tm.compute():.5f}\n"
                         f"accuracy\t{accuracy_tm.compute():.5f}\n"
                         f"precision\t{precision_tm.compute():.5f}\n"
                         f"f1\t\t{f1_tm.compute():.5f}\n"
                         f"prauc\t\t{prauc:.5f}\n"
                         f"sensitivity\t{sensitivity_tm.compute():.5f}\n"
                         f"specificity\t{specificity_tm.compute():.5f}\n"
                         f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")

            with open(f"output/eval/eegnet_"
                      f"{datetime.now().strftime('%y-%m-%d_%H_%M_%S')}.txt",
                      "w") as f:
                f.write(f"\nEvaluation (torch_metrics) results:"
                        f"\nmse\t\t{mse_tm.compute():.5f}\n"
                        f"roc\t\t{auroc_tm.compute():.5f}\n"
                        f"accuracy\t{accuracy_tm.compute():.5f}\n"
                        f"precision\t{precision_tm.compute():.5f}\n"
                        f"f1\t\t{f1_tm.compute():.5f}\n"
                        f"prauc\t\t{prauc:.5f}\n"
                        f"sensitivity\t{sensitivity_tm.compute():.5f}\n"
                        f"specificity\t{specificity_tm.compute():.5f}\n"
                        f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")
            # fig, ax = roc.plot(score=True)
            # fig.savefig(f"output/eval/lstm_"
            #             f"{datetime.now().strftime('%y-%m-%d_%H_%M_%S')}.png")
        if post_processing:
            post_processor = PostProcessor(y_true, y_pred)
            results_proc, _ = post_processor.metrics_with_drift()

            logging.info(
                f"\nEvaluation results (Post-processing):"
                f"\nmse\t\t{results_proc['mse']:.5f}\n"
                f"roc\t\t{results_proc['roc']:.5f}\n"
                f"accuracy\t{results_proc['accuracy']:.5f}\n"
                f"precision\t{results_proc['precision']:.5f}\n"
                f"f1\t\t{results_proc['f1']:.5f}\n"
                f"prauc\t\t{results_proc['prauc']:.5f}\n"
                f"sensitivity\t{results_proc['sensitivity']:.5f}\n"
                f"specificity\t{results_proc['specificity']:.5f}\n"
                f"TN: {results_proc['tn']} FP: {results_proc['fp']} "
                f"FN: {results_proc['fn']} TP: {results_proc['tp']}")

        return running_loss / len(data)


class EEGNetModel(nn.Module):
    def __init__(self, dev: device, config: dict = {
        'conv1_hd': 64,
        'conv1_k': 3,
        'conv2_k': 3,
        'avg_pool_x': 4,
        'avg_pool_y': 1,
        'f1_hd': 256,
        'weight_decay': 1e-3,
        'batch_size': 32,
        'frequency': 256,
        'channels': 18
    }):
        super(EEGNetModel, self).__init__()
        self.dev = dev
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.config['conv1_hd'],
                      padding='same',  # 'valid'/'same'/int/tuple(int,int)
                      kernel_size=self.config['conv1_k'],
                      bias=False),
            nn.BatchNorm2d(self.config['conv1_hd']),

            # DepthwiseConv2D
            nn.Conv2d(in_channels=self.config['conv1_hd'],
                      out_channels=self.config['conv1_hd'] * 2,
                      kernel_size=self.config['conv2_k'],
                      padding='same',  # 'valid'/'same'/int/tuple(int,int)
                      groups=self.config['conv1_hd'],
                      bias=False),

            nn.BatchNorm2d(self.config['conv1_hd'] * 2),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(self.config['avg_pool_y'],
                                      self.config['avg_pool_x']),
                         stride=1),
            nn.Dropout(p=0.25),

            # SeparableConv2D
            nn.Conv2d(in_channels=self.config['conv1_hd'] * 2,
                      out_channels=self.config['conv1_hd'] * 4,
                      padding='same',
                      kernel_size=(1, 1),
                      groups=self.config['conv1_hd'] * 2),

            nn.BatchNorm2d(self.config['conv1_hd'] * 4),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(self.config['avg_pool_y'],
                                      self.config['avg_pool_x'] * 2),
                         stride=1),
            nn.Dropout(p=0.25),
        )

        self.conv_out_size = (self.config['frequency'] - (
                self.config['avg_pool_x'] - 1 +
                self.config['avg_pool_x'] * 2 - 1))

        self.output = nn.Linear(in_features=(self.config['conv1_hd'] * 4 *
                                             self.conv_out_size *
                                             self.config['channels']),
                                out_features=2)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = x.view(-1, (self.config['conv1_hd'] * 4 *
                        self.conv_out_size *
                        self.config['channels']))

        return self.output(x)
