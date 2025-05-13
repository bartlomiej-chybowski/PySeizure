from datetime import datetime
import optuna
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import copy
import torchmetrics
from torch import amp
from typing import List, Dict

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pyseizure.classifier.classifier import Classifier
from pyseizure.data_classes.classifier import Classifier as classifier_name
from pyseizure.data_classes.objective import Objective
from pyseizure.helpers.metrics import get_confusion_matrix, get_specificity, \
    get_sensitivity, get_prauc, get_precision, get_accuracy, get_roc, \
    get_mse, get_f1, PostProcessor
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.helpers.supervised_contrastive_loss import \
    SupervisedContrastiveLoss
from pyseizure.helpers.torch_dataset import TorchDataset
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector
from sklearn.metrics import auc, RocCurveDisplay

VERBOSITY = 0


class LR(Classifier):
    def __init__(self, train_samples: List[str], valid_samples: List[str],
                 test_samples: List[str], normaliser: Normaliser = None,
                 feature_selector: FeatureSelector = None,
                 eval_only: bool = False, config: Dict = None):
        super().__init__(train_samples, test_samples, valid_samples,
                         normaliser, feature_selector, eval_only, config)

        self.config.update({
            'channels': 1 if self.binary else self.no_channels
        })
        if not self.config['hyperparameters']:
            self.config.update({
                'optimiser': 'Adam',
                'lr': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 128,
            })
        self._init_loaders_and_model()

    def __del__(self):
        del self.dataset

    def load_model(self, model_path: str = ''):
        self.model = torch.load(model_path, weights_only=False)

    def _init_loaders_and_model(self, reinit_keys: bool = True):
        eval_only = False if self.eval_only else True

        self.dataset = {
            'train': TorchDataset(self.train_samples,
                                  self.label_encoder,
                                  self.feature_selector,
                                  self.normaliser,
                                  batch_size=1,
                                  channels=self.config['channels'],
                                  use_scoring=self.use_scoring,
                                  subsampling=eval_only,
                                  binary=self.binary,
                                  artefacts=self.artefacts),
            'eval': TorchDataset(self.valid_samples,
                                 self.label_encoder,
                                 self.feature_selector,
                                 self.normaliser,
                                 batch_size=1,
                                 channels=self.config['channels'],
                                 use_scoring=self.use_scoring,
                                 subsampling=eval_only,
                                 binary=self.binary,
                                 artefacts=self.artefacts),
            'test': TorchDataset(self.test_samples,
                                 self.label_encoder,
                                 self.feature_selector,
                                 self.normaliser,
                                 batch_size=1,
                                 channels=self.config['channels'],
                                 use_scoring=False,
                                 subsampling=False,
                                 # subsampling=True,
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
        score_cols = 3
        if 'brain_state' in self.dataset['test'].all_columns:
            score_cols = 4
        self.n_features = len(self.dataset['test'].all_columns[:-score_cols])
        if self.feature_selector:
            x, _ = self.dataset['eval'].__getitem__(1)
            self.n_features = x.shape[1]

        self.model = LRModel(self.n_features).to(device=self.dev)
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

    def _train(self, trial):
        # w_neg = round(self.tune_weight[0], 3)
        # w_pos = round(self.tune_weight[1], 3)
        config = {
            'lr': trial.suggest_categorical(
                'lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
            'weight_decay': trial.suggest_categorical(
                'weight_decay', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            'batch_size': trial.suggest_int('batch_size', 32, 128, step=32),
        }
        model = LRModel(self.n_features).to(self.dev)
        train_dataset = TorchDataset(self.valid_samples,
                                     self.label_encoder,
                                     self.feature_selector,
                                     self.normaliser,
                                     batch_size=1,
                                     channels=self.config['channels'],
                                     use_scoring=self.use_scoring,
                                     subsampling=True,
                                     binary=self.binary,
                                     artefacts=self.artefacts)
        test_dataset = TorchDataset(self.test_samples,
                                    self.label_encoder,
                                    self.feature_selector,
                                    self.normaliser,
                                    batch_size=1,
                                    channels=self.config['channels'],
                                    use_scoring=self.use_scoring,
                                    subsampling=True,
                                    binary=self.binary,
                                    artefacts=self.artefacts)
        test_loader = DataLoader(test_dataset,
                                 shuffle=False,
                                 num_workers=self.no_threads_per_core,
                                 batch_size=self.config['batch_size'],
                                 drop_last=True,
                                 pin_memory=True)
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=self.no_threads_per_core,
                                  pin_memory=True,
                                  batch_size=self.config['batch_size'],
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
        f1 = torchmetrics.AveragePrecision(task="multiclass",
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
                    inputs = inputs.to(self.dev)
                    labels = labels.type(torch.LongTensor)
                    labels = labels[:, :, 0].flatten().to(self.dev)
                    output = model(inputs.float())
                    loss = criterion(output.squeeze(1), labels)

                if scheduler:
                    scheduler.step()
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                # loss.backward()
                # optimiser.step()

            model.eval()
            with torch.no_grad():
                for i, (data, labels) in enumerate(test_loader):
                    data = data.to(self.dev)
                    labels = labels.type(torch.LongTensor)
                    labels = labels[:, :, 0].flatten().to(self.dev)
                    output = model(data.float())
                    running_loss += criterion(output.squeeze(1),
                                              labels).item()
                    pred = softmax(output.squeeze(1))
                    f1.update(pred, labels)
                    prauc.update(pred, labels)
                    sensitivity.update(pred, labels)
                    specificity.update(pred, labels)
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
            study_name=f'{classifier_name.LOGISTIC_REGRESSION.value.lower()}'
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
                  f'{classifier_name.LOGISTIC_REGRESSION.value.lower()}'
                  f'_{self.dataset_name}.txt', 'w') as file:
            file.write(f'{str(self.train_samples)}\n')
            self.config['tune_objective'] = self.config['tune_objective'].value
            file.write(str(self.config))
        self._init_loaders_and_model(False)

        del study
        del trial

    def train(self):
        """
        Train LogisticRegression model.

        This function tunes hyperparameters, saves the best in the object
        variable, and trains model using them.

        Returns
        -------
        None
        """
        writer = SummaryWriter(
            f"output/runs/{classifier_name.LOGISTIC_REGRESSION.value.lower()}"
            f"_{self.dataset_name}_"
            f"{datetime.now().strftime('%y-%m-%d_%H_%M_%S')}")
        mse = torchmetrics.MeanSquaredError().to(self.dev)
        precision = torchmetrics.Precision(task='multiclass',
                                           num_classes=2,
                                           average='macro').to(self.dev)
        auroc = torchmetrics.AUROC(task='multiclass',
                                   num_classes=2).to(self.dev)
        roc = torchmetrics.ROC(task='multiclass',
                               num_classes=2).to(self.dev)
        f1 = torchmetrics.AveragePrecision(task='multiclass',
                                           num_classes=2).to(self.dev)
        prc = torchmetrics.PrecisionRecallCurve(task='multiclass',
                                                num_classes=2).to(self.dev)
        accuracy = torchmetrics.Accuracy(task='multiclass',
                                         num_classes=2).to(self.dev)
        specificity = torchmetrics.Specificity(task='multiclass',
                                               num_classes=2,
                                               average='macro').to(self.dev)
        sensitivity = torchmetrics.Recall(task='multiclass',
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
                pbar.set_description(f"Epoch {epoch+1:3d}", refresh=True)
                running_loss = 0.0
                f1.reset()
                prc.reset()
                mse.reset()
                precision.reset()
                auroc.reset()
                roc.reset()
                accuracy.reset()
                sensitivity.reset()

                for i, (data, labels) in enumerate(self.data_loader['train']):
                    # data = data.squeeze(0).to(self.dev)
                    self.optimiser.zero_grad()
                    with amp.autocast(device_type=self.dev.type, enabled=True):
                        data = data.to(self.dev)
                        labels = labels.type(torch.LongTensor)
                        labels = labels[:, :, 0].flatten().to(self.dev)

                        predictions = self.model(data)
                        predictions = predictions.squeeze(1)
                        loss = self.criterion(predictions, labels)
                        preds = softmax(predictions)
                        mse.update(preds[:, 1], labels)
                        precision.update(preds, labels)
                        auroc.update(preds, labels)
                        roc.update(preds, labels)
                        f1.update(preds, labels)
                        prc.update(preds, labels)
                        specificity.update(preds, labels)
                        sensitivity.update(preds, labels)
                        accuracy.update(preds, labels)

                    # loss.backward()
                    # self.optimiser.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimiser)
                    self.scaler.update()

                    running_loss += loss.item()
                    torch.cuda.empty_cache()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                    pbar.update(1)
                if i == 0:
                    i = 1
                if writer is not None:
                    writer.add_scalar("Loss/train",
                                      running_loss / ((epoch + 1) * i), epoch)
                    writer.add_scalar("auroc/train", auroc.compute(), epoch)
                    writer.add_scalar("accuracy/train", accuracy.compute(),
                                      epoch)
                    writer.add_scalar("f1/train", f1.compute(), epoch)
                    writer.add_scalar("prauc/train", auc(
                        prc.compute()[1][1].cpu().numpy(),
                        prc.compute()[0][1].cpu().numpy()
                    ), epoch)
                    writer.add_scalar("precision/train", precision.compute(),
                                      epoch)
                    writer.add_scalar("mse/train", mse.compute(), epoch)
                    writer.add_scalar("sensitivity/train",
                                      sensitivity.compute(), epoch)
                    writer.add_scalar("specificity/train",
                                      specificity.compute(), epoch)
                else:
                    fig, ax = roc.plot(score=True)
                    fig.savefig(
                        f"output/train/"
                        f"{classifier_name.LOGISTIC_REGRESSION.value.lower()}"
                        f"_{epoch}.png")
                    plt.close()
                    pbar.set_postfix(
                        loss=f"{running_loss / ((epoch + 1) * i):.5f}",
                        accuracy=f"{accuracy.compute():.5f}",
                        roc=f"{auroc.compute():.5f}",
                        sensitivity=f"{sensitivity.compute():.5f}",
                        specificity=f"{specificity.compute():.5f}",
                        f1=f"{f1.compute():.5f}",
                        precision=f"{precision.compute():.5f}",
                        mse=f"{mse.compute():.5f}",
                        refresh=True)

                if epoch % 5 == 0:
                    eval_loss = self.evaluate(self.data_loader['valid'], epoch,
                                              writer)
                    self.model.train()

                if best_eval_loss > eval_loss:
                    best_eval_loss, best_val_epoch = eval_loss, epoch
                    torch.save(
                        self.model,
                        f'output/models/'
                        f'{classifier_name.LOGISTIC_REGRESSION.value.lower()}'
                        f'_{self.dataset_name}'
                        f'_{self.feature_selector.name}.bin')

                if best_val_epoch < (epoch - patience):
                    break
            # plt.savefig(f'output/train/prauc_{epoch}_lr.png')
            # plt.close()

        self.model = torch.load(
            f'output/models/'
            f'{classifier_name.LOGISTIC_REGRESSION.value.lower()}'
            f'_{self.dataset_name}_{self.feature_selector.name}.bin',
            weights_only=False)

    def predict_eval(self, ax=None, fold: int = None,
                     plot_chance_level: bool = False):
        test_loader = DataLoader(self.dataset['test'], shuffle=False)
        data, y_true = next(iter(test_loader))
        data = data.to(self.dev)
        y_true = y_true.to(self.dev)
        predictions = self.model(data)
        acc = get_accuracy(predictions, y_true, data.shape[1])
        y_pred = predictions.cpu().squeeze(-1).detach().numpy().flatten()
        y_true = y_true.cpu().squeeze(-1).numpy().flatten()
        roc = get_roc(y_pred, y_true)
        prauc = get_prauc(y_pred, y_true)
        prec = get_precision(y_pred, y_true)
        mse = get_mse(y_pred[:, 1], y_true)
        tpr = get_sensitivity(y_pred, y_true)
        tnr = get_specificity(y_pred, y_true)

        viz = None
        if ax is not None and not np.isnan(roc):
            viz = RocCurveDisplay.from_predictions(
                y_true=y_true,
                y_pred=y_pred,
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
                plot_chance_level=plot_chance_level,
            )

        return y_pred, {
            'mse': mse,
            'roc': roc,
            'accuracy': acc,
            'precision': prec,
            'prauc': prauc,
            'sensitivity': tpr,
            'specificity': tnr
        }, viz

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
                labels = labels[:, :, 0].flatten().to(self.dev)
                inputs = inputs.squeeze(1).to(self.dev)
                outputs = self.model(inputs.float())
                preds = softmax(outputs)
                y_pred_tm = torch.cat((y_pred_tm, preds[:, 1]), 0)
                y_true_tm = torch.cat((y_true_tm, labels), 0)

        return y_pred_tm.cpu().numpy(), y_true_tm.cpu().numpy()

    def evaluate(self, dataloader: DataLoader = None,
                 epoch: int = None,
                 writer: SummaryWriter = None,
                 post_processing: bool = False) -> pd.DataFrame:
        """
        Evaluate model on test data.


        Function creates two figures:
            - most important features
            - the best iteration tree

        Returns
        -------
        pandas.DataFrame
            dataframe with each evaluation round
        """
        if dataloader is None:
            dataloader = self.data_loader['test']

        self.model.eval()
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
            for i, (data, labels) in enumerate(dataloader):
                data = data.to(self.dev)
                labels = labels.type(torch.LongTensor)
                labels = labels[:, :, 0].flatten().to(self.dev)
                predictions = self.model(data)
                predictions = predictions.squeeze(1)
                loss = self.criterion(predictions, labels)
                preds = softmax(predictions)
                running_loss += loss.item()
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
                              running_loss / len(dataloader), epoch)
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
            # fig, ax = roc.plot(score=True)
            # fig.savefig(f"output/eval/lstm_"
            #             f"{datetime.now().strftime('%y-%m-%d_%H_%M_%S')}.png")

        return running_loss / len(dataloader)

    def cross_validate(self, output_name: str):
        xgb_results = []
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(10, 6))

        for fold in range(len(self.train_samples)):
            logging.info(f'Train and evaluation, fold {fold}')
            train = copy(self.train_samples)
            train.pop(fold)
            self.train_dataset = TorchDataset(
                train,
                self.label_encoder,
                self.feature_selector,
                self.normaliser,
                channels=1 if self.binary else self.no_channels)
            self.test_dataset = TorchDataset(
                [self.train_samples[fold]],
                self.label_encoder,
                self.feature_selector,
                self.normaliser,
                channels=1 if self.binary else self.no_channels)

            self.train()
            y_pred, score, viz = self.predict_eval(
                ax, fold, (fold == len(self.train_samples) - 1))
            xgb_results.append(pd.DataFrame([score]))

            if viz:
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Mean ROC curve with variability",
        )
        ax.axis("square")
        plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5),
                   fancybox=True, shadow=True, ncol=1)
        plt.subplots_adjust(right=0.7)
        plt.savefig(f'output/{output_name}.png')

        results = pd.concat(xgb_results, ignore_index=True)
        logging.info(results, results.mean(axis=0))
        results.to_csv(f'output/{output_name}.csv')

        self.train_dataset = TorchDataset(
            self.train_samples,
            self.label_encoder,
            self.feature_selector,
            self.normaliser,
            channels=1 if self.binary else self.no_channels)
        self.test_dataset = TorchDataset(
            self.test_samples,
            self.label_encoder,
            self.feature_selector,
            self.normaliser,
            channels=1 if self.binary else self.no_channels)


class LRModel(nn.Module):
    def __init__(self, n_features):
        super(LRModel, self).__init__()
        self.linear = torch.nn.Linear(n_features, 2)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='leaky_relu')
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # return torch.sigmoid(self.linear(x))
        return self.linear(x)
