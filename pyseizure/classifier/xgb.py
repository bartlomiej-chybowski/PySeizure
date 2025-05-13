import gc
import optuna
import torch
import logging
import numpy as np
import pandas as pd
import torchmetrics
import xgboost as xgb
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from copy import copy
from torch import cuda
from datetime import datetime
from typing import List, Dict
from torch.utils.tensorboard import SummaryWriter
from pyseizure.data_classes.objective import Objective
from pyseizure.helpers.iterator import Iterator
from sklearn.metrics import auc
from pyseizure.helpers.normaliser import Normaliser
from pyseizure.classifier.classifier import Classifier
from pyseizure.data_classes.classifier import Classifier as classifier_name
from pyseizure.helpers.metrics import get_f1, get_prauc, get_sensitivity, \
    get_specificity, get_confusion_matrix, PostProcessor
from pyseizure.preprocessing.feature_selection.feature_selector import \
    FeatureSelector


class XGB(Classifier):
    def __init__(self, train_samples: List[str], valid_samples: List[str],
                 test_samples: List[str], normaliser: Normaliser = None,
                 feature_selector: FeatureSelector = None,
                 eval_only: bool = False, config: Dict = None):
        super().__init__(train_samples, test_samples, valid_samples,
                         normaliser, feature_selector, eval_only, config)
        self.model = None
        num_class = self.label_encoder.classes_.size
        self.config.update({
            'seed': 42,
            'verbosity': 0,
            'device': 'cuda' if cuda.is_available() else 'cpu',
            'num_class': None if self.binary else num_class,
        })
        if not self.config['hyperparameters']:
            self.config.update({
                'n_jobs': self.no_threads_per_core,
                'n_estimators': 200,
                'num_parallel_tree': 50,
                'learning_rate': 0.01,
                'eval_metric': ['logloss', 'auc', 'aucpr'],
                'tree_method': 'hist',
                'max_bin': 256,
                'gamma': 0.01,
                'lambda': 0.1,
                'alpha': 0.1,
                # 'max_delta_step': 1,
                'early_stopping_rounds': 50,
                'batch_size': 256,
                'objective': 'binary:hinge' if self.binary else 'multi:softmax'
            })
        self._init_loaders_and_model()

    def load_model(self, model_path: str = ''):
        self.model = xgb.Booster({'nthread': -1})
        self.model.load_model(model_path)

    def _init_loaders_and_model(self, reinit_keys: bool = True):
        self.data_loader = {
            'train': Iterator(self.train_samples,
                              self.label_encoder,
                              self.feature_selector,
                              self.normaliser,
                              batch_size=self.config['batch_size'],
                              evaluation=self.eval_only,
                              use_scoring=self.use_scoring,
                              subsampling=not self.eval_only,
                              shuffle=not self.eval_only,
                              binary=self.binary,
                              artefacts=self.artefacts),
            'valid': Iterator(self.valid_samples,
                              self.label_encoder,
                              self.feature_selector,
                              self.normaliser,
                              batch_size=self.config['batch_size'],
                              evaluation=self.eval_only,
                              use_scoring=self.use_scoring,
                              subsampling=not self.eval_only,
                              shuffle=not self.eval_only,
                              binary=self.binary,
                              artefacts=self.artefacts),
            'test': Iterator(self.test_samples,
                             self.label_encoder,
                             self.feature_selector,
                             self.normaliser,
                             subsampling=False,
                             # subsampling=True,
                             batch_size=self.config['batch_size'],
                             evaluation=True,
                             use_scoring=False,
                             shuffle=False,
                             binary=self.binary,
                             artefacts=self.artefacts)
        }

    def _train(self, trial):
        params = copy(self.config)

        it_valid = Iterator(self.valid_samples,
                            self.label_encoder,
                            self.feature_selector,
                            self.normaliser,
                            batch_size=self.config['batch_size'],
                            evaluation=False,
                            subsampling=True,
                            use_scoring=self.use_scoring,
                            shuffle=True,
                            binary=self.binary,
                            artefacts=self.artefacts)
        it_test = Iterator(self.test_samples,
                           self.label_encoder,
                           self.feature_selector,
                           self.normaliser,
                           batch_size=self.config['batch_size'],
                           evaluation=False,
                           subsampling=True,
                           use_scoring=False,
                           shuffle=False,
                           binary=self.binary,
                           artefacts=self.artefacts)

        xy = xgb.QuantileDMatrix(it_valid, missing=np.nan,
                                 enable_categorical=False)
        xy_eval = xgb.QuantileDMatrix(it_test, missing=np.nan,
                                      enable_categorical=False)

        max_depth = np.logspace(0.7, 1.6, 10, dtype='int').round(
            decimals=4).tolist()
        min_child_weight = np.logspace(0.5, 4, 10, dtype='int').round(
            decimals=4).tolist()
        reg_alpha = np.geomspace(0.001, 10, 10, dtype='float').round(
            decimals=4).tolist()
        reg_lambda = np.geomspace(0.001, 10, 10, dtype='float').round(
            decimals=4).tolist()
        reg_gamma = np.geomspace(0.000001, 0.2, 10, dtype='float').round(
            decimals=4).tolist()
        max_delta_step = np.geomspace(0.1, 10, 10, dtype='float').round(
            decimals=4).tolist()
        learning_rate = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
        subsample = [0.55, 0.6, 0.70, 0.85]
        colsample_bytree = np.geomspace(0.3, 1.0, 10, dtype='float').round(
            decimals=4).tolist()
        # num_parallel_tree = np.geomspace(200, 1100, 9, dtype='int').round(
        #     decimals=4).tolist()
        num_parallel_tree = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        # n_estimators = np.geomspace(100, 10000, 10, dtype='int').round(
        #     decimals=4).tolist()
        n_estimators = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        params.update({
            'n_estimators': trial.suggest_categorical('n_estimators',
                                                      n_estimators),
            'learning_rate': trial.suggest_categorical('learning_rate',
                                                       learning_rate),
            'max_depth': trial.suggest_categorical('max_depth', max_depth),
            'max_delta_step': trial.suggest_categorical('max_delta_step',
                                                        max_delta_step),
            'min_child_weight': trial.suggest_categorical('min_child_weight',
                                                          min_child_weight),
            'gamma': trial.suggest_categorical('gamma', reg_gamma),
            'lambda': trial.suggest_categorical('lambda', reg_lambda),
            'alpha': trial.suggest_categorical('alpha', reg_alpha),
            'subsample': trial.suggest_categorical('subsample', subsample),
            'num_parallel_tree': trial.suggest_categorical('num_parallel_tree',
                                                           num_parallel_tree),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                          colsample_bytree),
            'base_score': it_valid.base_score,
            'scale_pos_weight': it_valid.weights,
            'batch_size': trial.suggest_int('batch_size', 32, 256, step=32),
        })

        model = xgb.train(
            params,
            xy,
            num_boost_round=params['n_estimators'],
            early_stopping_rounds=params['early_stopping_rounds'],
            evals=[(xy, 'train'), (xy_eval, 'valid')],
            verbose_eval=True)
        y_pred = model.predict(xy_eval, output_margin=False)
        y_true = it_test.get_y_true()[:len(y_pred)]

        f1 = get_f1(y_pred, y_true)
        prauc = get_prauc(y_pred, y_true)
        sensitivity = get_sensitivity(y_pred, y_true)
        specificity = get_specificity(y_pred, y_true)
        tn, fp, fn, tp = get_confusion_matrix(y_pred.round(), y_true)

        trial.set_user_attr("CM", f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        trial.set_user_attr("PRAUC", f"{prauc:.5f}")
        trial.set_user_attr("F1", f"{f1:.5f}")
        trial.set_user_attr("Sensitivity", f"{sensitivity:.5f}")
        trial.set_user_attr("Specificity", f"{specificity:.5f}")

        del xy
        del xy_eval
        del y_pred
        del y_true

        if self.config['tune_objective'] == Objective.FPFN:
            trial.report(fp + fn, 0)

            return fp + fn
        else:
            trial.report((f1 + sensitivity + specificity) / 3, 0)

            return (f1 + sensitivity + specificity) / 3

    def tune(self, config: Dict):
        direction = "minimize"
        if config['tune_objective'] == Objective.SSF:
            direction = "maximize"

        self.config['tune_objective'] = config['tune_objective']

        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            study_name=f'{classifier_name.XGBOOST.value.lower()}'
                       f'_{self.dataset_name}'
                       f'_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}',
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
        with open(f'output/models/{classifier_name.XGBOOST.value.lower()}'
                  f'_{self.dataset_name}.txt', 'w') as file:
            file.write(f'{str(self.train_samples)}\n')
            self.config['tune_objective'] = self.config['tune_objective'].value
            file.write(str(self.config))
        self._init_loaders_and_model(False)

        del study
        del trial

    def train(self):
        """
        Train XGBoost model.

        Returns
        -------
        None
        """
        xy = xgb.QuantileDMatrix(self.data_loader['train'], missing=np.nan,
                                 enable_categorical=False)

        xy_eval = xgb.QuantileDMatrix(self.data_loader['valid'],
                                      missing=np.nan, enable_categorical=False)
        self.config.update({
            'base_score': self.data_loader['train'].base_score,
            'scale_pos_weight': self.data_loader['train'].weights,
        })

        self.model = xgb.train(
            self.config,
            xy,
            num_boost_round=self.config['n_estimators'],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            evals=[(xy, 'train'), (xy_eval, 'valid')],
            verbose_eval=True)

        self.model.save_model(f'output/models/'
                              f'{classifier_name.XGBOOST.value.lower()}'
                              f'_{self.dataset_name}'
                              f'_{self.feature_selector.name}.bin')

        logging.debug(self.model)

    def predict(self, data: Iterator = None,
                return_proba: bool = True) -> np.array:
        """
        Predict output for provided data.

        Parameters
        ----------
        data: pandas.DataFrame
            input data
        return_proba: bool
            a flag: if true returns probability of each class instead of
            prediction

        Returns
        -------
        numpy.array
            numpy array with predictions and true labels of each class
        """
        if data is None:
            data = self.data_loader['test']

        y_pred = self.model.predict(
            xgb.QuantileDMatrix(data=data),
            output_margin=False if self.binary else return_proba)
        y_true = data.get_y_true()[:len(y_pred)]

        return y_pred, y_true

    def evaluate(self,
                 dataloader: Iterator = None,
                 epoch: int = None,
                 writer: SummaryWriter = None,
                 post_processing: bool = False) -> Dict:
        """
        Evaluate model on test data.

        Function creates two figures:
            - most important features
            - the best iteration tree

        Returns
        -------
        Dict
            dict with evaluation metrics
        """
        if dataloader is None:
            dataloader = self.data_loader['test']
        try:
            fig, ax = plt.subplots(figsize=(100, 50), dpi=300)
            xgb.plot_tree(self.model, num_trees=self.model.best_iteration,
                          fontsize=15, ax=ax)
            plt.savefig(f'output/{classifier_name.XGBOOST.value.lower()}'
                        f'_{self.dataset_name}_tree.png')

            fig, ax = plt.subplots(figsize=(25, 25), dpi=300)
            xgb.plot_importance(self.model, ax=ax)
            plt.savefig(f'output/{classifier_name.XGBOOST.value.lower()}'
                        f'_{self.dataset_name}_features.png')
        except ValueError as e:
            logging.info(e)

        y_pred = self.model.predict(xgb.QuantileDMatrix(data=dataloader),
                                    output_margin=False)
        y_pred_multi = np.concatenate([1 - y_pred.reshape(-1, 1),
                                       y_pred.reshape(-1, 1)], axis=1)
        y_true = dataloader.get_y_true()[: len(y_pred)]

        mse_tm = torchmetrics.MeanSquaredError()
        auroc_tm = torchmetrics.AUROC(task='multiclass', num_classes=2)
        roc_tm = torchmetrics.ROC(task='multiclass', num_classes=2)
        precision_tm = torchmetrics.Precision(task='multiclass',
                                              num_classes=2,
                                              average='macro')
        f1_tm = torchmetrics.AveragePrecision(task='multiclass', num_classes=2)
        prc_tm = torchmetrics.PrecisionRecallCurve(task='multiclass',
                                                   num_classes=2)
        accuracy_tm = torchmetrics.Accuracy(task='multiclass', num_classes=2)
        specificity_tm = torchmetrics.Specificity(task='multiclass',
                                                  num_classes=2,
                                                  average='macro')
        sensitivity_tm = torchmetrics.Recall(task='multiclass', num_classes=2)
        confmat = torchmetrics.ConfusionMatrix(task='multiclass',
                                               num_classes=2)

        roc_tm.update(torch.tensor(y_pred_multi), torch.tensor(y_true))
        prc_tm.update(torch.tensor(y_pred_multi), torch.tensor(y_true))
        precision_tm.update(torch.tensor(y_pred_multi), torch.tensor(y_true))
        mse_tm.update(torch.tensor(y_pred), torch.tensor(y_true))
        accuracy_tm.update(torch.tensor(y_pred_multi), torch.tensor(y_true))
        auroc_tm.update(torch.tensor(y_pred_multi), torch.tensor(y_true))
        f1_tm.update(torch.tensor(y_pred_multi), torch.tensor(y_true))
        specificity_tm.update(torch.tensor(y_pred_multi), torch.tensor(y_true))
        sensitivity_tm.update(torch.tensor(y_pred_multi), torch.tensor(y_true))
        cm = confmat(torch.tensor(y_pred_multi).round(), torch.tensor(y_true))
        pr_auc = auc(prc_tm.compute()[1][1].numpy(),
                     prc_tm.compute()[0][1].numpy())
        tn, fp, fn, tp = cm.numpy().ravel()

        logging.info(f"\nEvaluation (torch_metrics) results:"
                     f"\nmse\t\t{mse_tm.compute():.5f}\n"
                     f"roc\t\t{auroc_tm.compute():.5f}\n"
                     f"accuracy\t{accuracy_tm.compute():.5f}\n"
                     f"precision\t{precision_tm.compute():.5f}\n"
                     f"f1\t\t{f1_tm.compute():.5f}\n"
                     f"prauc\t\t{pr_auc:.5f}\n"
                     f"sensitivity\t{sensitivity_tm.compute():.5f}\n"
                     f"specificity\t{specificity_tm.compute():.5f}\n"
                     f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")

        # fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        # _, score, _ = self.predict(dataloader, True, ax=ax)
        # y_pred = self.model.predict(xy, True)

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

        return {
            'mse': mse_tm.compute(),
            'roc': roc_tm.compute(),
            'f1': f1_tm.compute(),
            'accuracy': accuracy_tm.compute(),
            'precision': precision_tm.compute(),
            'prauc': pr_auc,
            'sensitivity': sensitivity_tm.compute(),
            'specificity': specificity_tm.compute(),
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }

    def cross_validate(self, output_name: str):
        original_train = copy(self.train_samples)
        original_test = copy(self.test_samples)

        xgb_results = []
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(10, 6))

        for fold in range(len(original_train)):
            logging.info(f'Train and evaluation, fold {fold}')
            train = copy(original_train)
            train.pop(fold)
            self.train_samples = train
            self.test_samples = [original_test[fold]]
            self.train()
            y_pred, score, viz = self.predict(
                pq.read_table(original_test[fold]).to_pandas(),
                True, ax, fold, (fold == len(original_train) - 1))
            xgb_results.append(pd.DataFrame([score]))

            if viz:
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            self.model.__del__()
            torch.cuda.empty_cache()
            gc.collect()

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

        self.train_samples = original_train
        self.test_samples = original_test
