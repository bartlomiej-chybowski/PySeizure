import math

import torch
import optuna
import logging
import numpy as np
import torchmetrics
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler
from tqdm import tqdm
from datetime import datetime
from torch import device, amp
from sklearn.metrics import auc
from torch.utils.data import DataLoader
from typing import List, Dict
from torch.utils.tensorboard import SummaryWriter

from pyseizure.data_classes.objective import Objective
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.classifier.classifier import Classifier
from pyseizure.data_classes.classifier import Classifier as classifier_name
from pyseizure.helpers.conv_dataset import ConvDataset
from pyseizure.helpers.supervised_contrastive_loss import \
    SupervisedContrastiveLoss
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector
from pyseizure.helpers.metrics import get_confusion_matrix, get_mse, get_roc, \
    get_accuracy, get_precision, get_f1, get_prauc, get_sensitivity, \
    get_specificity, PostProcessor

VERBOSITY = 0


class ConvTransformer(Classifier):
    def __init__(self, train_samples: List[str], valid_samples: List[str],
                 test_samples: List[str], normaliser: Normaliser = None,
                 feature_selector: FeatureSelector = None,
                 eval_only: bool = False, config: Dict = None):
        super().__init__(train_samples, test_samples, valid_samples,
                         normaliser, feature_selector, eval_only, config)
        self.config.update({
            'channels': self.no_channels,
            'frequency': 256,
        })
        if not self.config['hyperparameters']:
            self.config.update({
                'conv1_hd': 512,
                'conv1_k': 5,
                'max_pool': 2,
                'conv2_hd': 256,
                'conv2_k': 4,
                'conv3_hd': 128,
                'conv3_k': 3,
                'vocab_size': 2100,
                'feedforward': 128,
                'n_head': 4,
                'n_layers': 2,
                'd_model': 100,
                'lr': 1e-5,
                'weight_decay': 1e-2,
                'batch_size': 64,
                'optimiser': 'Adam'
            })

        self._init_loaders_and_model()

    def _init_loaders_and_model(self, reinit_keys: bool = True):
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
        self.model = ConvTransformerModel(self.dev, self.config).to(self.dev)

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

    def _train(self, trial):
        d_model = trial.suggest_categorical('d_model',
                                            [100, 200, 300, 500, 800])
        n_head = trial.suggest_categorical('n_head', [2, 3, 4, 5, 6])
        if d_model % n_head != 0:
            raise optuna.exceptions.TrialPruned()

        config = {
            'conv1_hd': trial.suggest_categorical(
                'conv1_hd', [2**i for i in range(8, 11)]),
            'conv1_k': trial.suggest_int('conv1_k', 3, 5),
            'max_pool': 2,
            'conv2_hd': trial.suggest_categorical(
                'conv2_hd', [2**i for i in range(7, 9)]),
            'conv2_k': trial.suggest_int('conv2_k', 3, 5),
            'conv3_hd': trial.suggest_categorical(
                'conv3_hd', [2**i for i in range(6, 8)]),
            'conv3_k': trial.suggest_int('conv3_k', 3, 5),
            'vocab_size': trial.suggest_categorical(
                'vocab_size', [1300, 2100, 3400, 5500, 8900]),
            'feedforward': trial.suggest_categorical(
                'feedforward', [2 ** i for i in range(7, 11)]),
            'd_model': d_model,
            'n_head': n_head,
            'n_layers': trial.suggest_categorical(
                'n_layers', [1, 2, 3, 4, 5, 6]),
            'lr': trial.suggest_categorical(
                'lr', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]),
            'weight_decay': trial.suggest_categorical(
                'weight_decay', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
            'batch_size': trial.suggest_int('batch_size', 32, 128, step=32),
            'frequency': 256,
            'channels': self.no_channels
        }
        model = ConvTransformerModel(self.dev, config).to(self.dev)

        test_dataset = ConvDataset(self.test_samples,
                                   self.label_encoder,
                                   self.feature_selector,
                                   self.normaliser,
                                   batch_size=config['batch_size'],
                                   channels=config['channels'],
                                   use_scoring=self.use_scoring,
                                   subsampling=True,
                                   binary=self.binary,
                                   artefacts=self.artefacts)
        train_dataset = ConvDataset(self.valid_samples,
                                    self.label_encoder,
                                    self.feature_selector,
                                    self.normaliser,
                                    batch_size=config['batch_size'],
                                    channels=config['channels'],
                                    use_scoring=self.use_scoring,
                                    subsampling=True,
                                    binary=self.binary,
                                    artefacts=self.artefacts)
        test_loader = DataLoader(test_dataset,
                                 shuffle=False,
                                 num_workers=self.no_threads_per_core,
                                 batch_size=config['batch_size'],
                                 drop_last=True,
                                 pin_memory=True)
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=self.no_threads_per_core,
                                  pin_memory=True,
                                  batch_size=config['batch_size'],
                                  drop_last=True)
        if self.use_contrastive:
            criterion = SupervisedContrastiveLoss(temperature=.1)
        else:
            criterion = nn.CrossEntropyLoss()
        optimiser_name = trial.suggest_categorical("optimiser",
                                                   ["Adam", "AdamW", "SGD"])
        optimiser = getattr(torch.optim, optimiser_name)(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'])
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimiser,
            T_max=5 * len(train_loader),
            eta_min=config['lr'])
        scaler = amp.GradScaler(self.dev.type)

        f1 = torchmetrics.AveragePrecision(task='multiclass',
                                           num_classes=2).to(self.dev)
        specificity = torchmetrics.Specificity(task='multiclass',
                                               num_classes=2,
                                               average='macro').to(self.dev)
        sensitivity = torchmetrics.Recall(task='multiclass',
                                          num_classes=2).to(self.dev)
        prauc = torchmetrics.PrecisionRecallCurve(task='multiclass',
                                                  num_classes=2).to(self.dev)
        confmat = torchmetrics.ConfusionMatrix(task='multiclass',
                                               num_classes=2).to(self.dev)
        softmax = nn.Softmax(dim=1)
        # len_test = len(test_loader)
        best_y_pred = torch.Tensor().to(self.dev)
        best_y_true = torch.Tensor().to(self.dev)
        best_result = 0
        if self.config['tune_objective'] == Objective.FPFN:
            best_result = 10000000
        best_f1, best_prauc, best_sensitivity, best_specificity = 0, 0, 0, 0
        for epoch in range(self.config['n_epochs']):
            y_pred = torch.Tensor().to(self.dev)
            y_true = torch.Tensor().to(self.dev)
            f1.reset()
            prauc.reset()
            sensitivity.reset()
            specificity.reset()
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimiser.zero_grad()
                with amp.autocast(device_type=self.dev.type, enabled=True):
                    labels = labels.type(torch.LongTensor)
                    inputs = inputs.squeeze(0).to(self.dev)
                    labels = labels.to(self.dev)
                    output = model(inputs.float())
                    loss = criterion(output, labels)

                if scheduler:
                    scheduler.step()
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                # loss.backward()
                # optimiser.step()

            model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(test_loader):
                    labels = labels.type(torch.LongTensor)
                    inputs = inputs.squeeze(0).to(self.dev)
                    labels = labels.to(self.dev)
                    output = model(inputs.float())
                    running_loss += criterion(output, labels).item()
                    pred = softmax(output)
                    f1.update(pred, labels)
                    prauc.update(pred, labels)
                    sensitivity.update(pred, labels)
                    specificity.update(pred, labels)
                    # cm = confmat(y_pred.round(), y_true)
                    # tn_train, _, _, tp_train = cm.cpu().numpy().ravel()
                    y_pred = torch.cat((y_pred, pred[:, 1]), 0)
                    y_true = torch.cat((y_true, labels), 0)

            confmat.reset()
            _, fp, fn, _ = confmat(y_pred.round(),
                                   y_true).cpu().numpy().ravel()
            if self.config['tune_objective'] == Objective.FPFN:
                result = fp + fn
                # optimise and extract method
                if best_result > result:
                    best_result = result
                    best_y_pred = y_pred
                    best_y_true = y_true
                    best_prauc = prauc
                    best_sensitivity = sensitivity
                    best_specificity = specificity
                    best_f1 = f1
            else:
                result = (f1.compute() + sensitivity.compute() +
                          specificity.compute()) / 3
                if best_result < result:
                    best_result = result
                    best_y_pred = y_pred
                    best_y_true = y_true
                    best_prauc = prauc
                    best_sensitivity = sensitivity
                    best_specificity = specificity
                    best_f1 = f1
            trial.report(result, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # torch.cuda.empty_cache()
        confmat.reset()
        cm = confmat(best_y_pred.round(), best_y_true)
        tn, fp, fn, tp = cm.cpu().numpy().ravel()
        prauc = auc(best_prauc.compute()[1][1].cpu().numpy(),
                    best_prauc.compute()[0][1].cpu().numpy())
        trial.set_user_attr("CM", f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        trial.set_user_attr("PRAUC", f"{prauc:.5f}")
        trial.set_user_attr("F1", f"{best_f1.compute():.5f}")
        trial.set_user_attr("Sensitivity", f"{best_sensitivity.compute():.5f}")
        trial.set_user_attr("Specificity", f"{best_specificity.compute():.5f}")

        del test_loader
        del train_loader
        del model
        del criterion
        del optimiser
        del config
        del y_pred
        del y_true
        del best_y_pred
        del best_y_true

        return best_result

    def tune(self, config: Dict):
        direction = "minimize"
        if config['tune_objective'] == Objective.SSF:
            direction = "maximize"

        self.config['tune_objective'] = config['tune_objective']
        self.config['n_epochs'] = config['n_epochs']
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            study_name=f'{classifier_name.CONV_TRANSFORMER.value.lower()}'
                       f'_{self.dataset_name}_'
                       f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}',
            direction=direction,
            sampler=config['sampler'],
            pruner=config['pruner'])

        study.optimize(self._train,
                       n_trials=config['n_trials'],
                       timeout=None,
                       gc_after_trial=True,
                       n_jobs=config['n_jobs'],
                       show_progress_bar=True)

        trial = study.best_trial
        logging.info(f"Best trial: {trial.value}")
        logging.info(f"Params: \n{trial.params}")

        self.config.update(trial.params)
        with open(f'output/models/'
                  f'{classifier_name.CONV_TRANSFORMER.value.lower()}'
                  f'_{self.dataset_name}.txt', 'w') as file:
            file.write(f'{str(self.train_samples)}\n')
            self.config['tune_objective'] = self.config['tune_objective'].value
            file.write(str(self.config))
        self._init_loaders_and_model()

        del study
        del trial

    @staticmethod
    def plot_single_epoch(inputs: torch.Tensor, labels: torch.Tensor):
        for i in range(inputs.shape[1]):
            plt.figure(figsize=(10, 5))
            plt.imshow(inputs[0][i].numpy(), interpolation='nearest',
                       aspect='auto')
            plt.title(f"Epoch: {i}, class: {labels[0][i][0][0].numpy()}")
            plt.tight_layout()
            plt.show()

    def train(self):
        writer = SummaryWriter(
            f"output/runs/{classifier_name.CONV_TRANSFORMER.value.lower()}"
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
                    fig.savefig(f"output/train/transformer_{epoch}.png")
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
                    torch.save(
                        self.model,
                        f'output/models/'
                        f'{classifier_name.CONV_TRANSFORMER.value.lower()}'
                        f'_{self.dataset_name}.bin')

                if best_val_epoch < (epoch - patience):
                    break

        self.model = torch.load(
            f'output/models/{classifier_name.CONV_TRANSFORMER.value.lower()}'
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


"""
Acknowledgements
================
This Transformer implementation is adapted for the needs of this project and is based on 
Transformer (https://github.com/hyunwoongko/transformer/tree/master) by Kevin Ko (hyunwoongko)
"""
class LayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-12

    def forward(self, x):
        out = x - x.mean(-1, keepdim=True)
        out /= torch.sqrt(x.var(-1, correction=0, keepdim=True) + self.eps)
        out = self.alpha * out + self.bias

        return out


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        batch_size, head, length, d_tensor = key.size()
        score = (query @ key.transpose(2, 3)) / math.sqrt(d_tensor)
        score = self.softmax(score)

        return (score @ value), score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        return tensor.view(batch_size, length, self.n_head,
                           d_model // self.n_head).transpose(1, 2)


    def forward(self, query, key, value):
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)
        query, key, value = (self.split(query), self.split(key),
                             self.split(value))

        x, attention = self.attention(query, key, value)

        batch_size, head, length, d_tensor = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, length,
                                                head * d_tensor)
        x = self.w_o(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.25):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, feed_forward, n_head, dropout):
        super(EncoderLayer, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm1 = LayerNorm(d_model=d_model)
        self.norm2 = LayerNorm(d_model=d_model)
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.feed_forward = FeedForward(d_model=d_model, hidden=feed_forward,
                                        drop_prob=dropout)

    def forward(self, x):
        _x = x
        x = self.attention(query=x, key=x, value=x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x


class ClassificationHead(nn.Module):
    def __init__(self, d_model, seq_len, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Flatten(),
            nn.Linear(d_model * seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.linear(x)


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, dropout):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, feed_forward=ffn_hidden,
                          n_head=n_head, dropout=dropout)
             for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dropout: float = 0.1, vocab_size: int = 5000,
                 d_model: int = 512):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(vocab_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))

        pe = torch.zeros(vocab_size, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)

        return x


class ConvTransformerModel(nn.Module):
    def __init__(self, dev: device, config: dict = {
        'frequency': 256,
        'conv1_hd': 512,
        'conv1_k': 5,
        'max_pool': 2,
        'conv2_hd': 256,
        'conv2_k': 4,
        'conv3_hd': 128,
        'conv3_k': 3,
        'vocab_size': 2100,
        'feedforward': 128,
        'n_head': 4,
        'n_layers': 2,
        'd_model': 100,
    }):
        super(ConvTransformerModel, self).__init__()
        self.dev = dev
        self.config = config

        conv1 = nn.Conv1d(in_channels=self.config['channels'],
                          out_channels=self.config['conv1_hd'],
                          kernel_size=self.config['conv1_k'])
        nn.init.kaiming_normal_(conv1.weight, nonlinearity='leaky_relu')

        conv2 = nn.Conv1d(in_channels=self.config['conv1_hd'],
                          out_channels=self.config['conv2_hd'],
                          kernel_size=self.config['conv2_k'])
        nn.init.kaiming_normal_(conv2.weight, nonlinearity='leaky_relu')

        conv3 = nn.Conv1d(in_channels=self.config['conv2_hd'],
                          out_channels=self.config['conv3_hd'],
                          kernel_size=self.config['conv3_k'])
        nn.init.kaiming_normal_(conv3.weight, nonlinearity='leaky_relu')

        self.conv = nn.Sequential(
            conv1,
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.config['conv1_hd']),

            nn.MaxPool1d(kernel_size=self.config['max_pool'], stride=1),
            nn.Dropout(p=0.5),

            conv2,
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.config['conv2_hd']),

            conv3,
            nn.BatchNorm1d(self.config['conv3_hd']),
            nn.LeakyReLU(),

            nn.Dropout(p=0.5),
        )

        self.conv_out_size = (self.config['frequency'] - (
                self.config['conv1_k'] - 1 +
                self.config['conv2_k'] - 1 +
                self.config['conv3_k'] - 1 +
                self.config['max_pool'] - 1))

        linear = nn.Linear(in_features=(self.config['conv3_hd'] *
                                        self.conv_out_size),
                           out_features=self.config['frequency'])
        nn.init.kaiming_normal_(linear.weight, nonlinearity='leaky_relu')
        self.fc1 = nn.Sequential(
            linear,
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(self.config['frequency'])
        )

        self.encoder_input_layer = nn.Linear(
            in_features=1,
            out_features=self.config['d_model']
        )

        self.pos_emb = PositionalEncoding(vocab_size=self.config['vocab_size'],
                                          d_model=self.config['d_model'],
                                          dropout=0.25)
        self.encoder = Encoder(d_model=self.config['d_model'],
                               n_head=self.config['n_head'],
                               ffn_hidden=self.config['d_model'],
                               dropout=0.25,
                               n_layers=self.config['n_layers'])
        self.classHead = ClassificationHead(seq_len=self.config['frequency'],
                                            d_model=self.config['d_model'],
                                            n_classes=2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.config['conv3_hd'] * self.conv_out_size)
        x = self.fc1(x).unsqueeze(2)
        x = self.encoder_input_layer(x)
        x = self.pos_emb(x)
        x = self.encoder(x)
        x = self.classHead(x)

        return x
