import os
import shutil
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, TQDMProgressBar, ModelCheckpoint


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []
    logger = logger

    # Rich progress can fail in Colab notebooks with "pop from empty list".
    progress_bar = str(cfg.get('PROGRESS_BAR', 'rich')).lower()
    if progress_bar == 'tqdm':
        callbacks.append(TQDMProgressBar())
    elif progress_bar not in ['none', 'false', '0']:
        callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))
        
    return callbacks

def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
        "how2sign_DTW_MPJPE_PA_lhand": "Metrics/how2sign_DTW_MPJPE_PA_lhand",
        "how2sign_DTW_MPJPE_PA_rhand": "Metrics/how2sign_DTW_MPJPE_PA_rhand",
        "how2sign_DTW_MPJPE_PA_body": "Metrics/how2sign_DTW_MPJPE_PA_body",
        "csl_DTW_MPJPE_PA_lhand": "Metrics/csl_DTW_MPJPE_PA_lhand",
        "csl_DTW_MPJPE_PA_rhand": "Metrics/csl_DTW_MPJPE_PA_rhand",
        "csl_DTW_MPJPE_PA_body": "Metrics/csl_DTW_MPJPE_PA_body",
        "phoenix_DTW_MPJPE_PA_lhand": "Metrics/phoenix_DTW_MPJPE_PA_lhand",
        "phoenix_DTW_MPJPE_PA_rhand": "Metrics/phoenix_DTW_MPJPE_PA_rhand",
        "phoenix_DTW_MPJPE_PA_body": "Metrics/phoenix_DTW_MPJPE_PA_body",
        "how2sign_MPVPE_PA_all": "Metrics/how2sign_MPVPE_PA_all",
        "how2sign_MPJPE_PA_hand": "Metrics/how2sign_MPJPE_PA_hand",
        "csl_MPVPE_PA_all": "Metrics/csl_MPVPE_PA_all",
        "csl_MPJPE_PA_hand": "Metrics/csl_MPJPE_PA_hand",
        "phoenix_MPVPE_PA_all": "Metrics/phoenix_MPVPE_PA_all",
        "phoenix_MPJPE_PA_hand": "Metrics/phoenix_MPJPE_PA_hand",
        "BLEU_1": "Metrics/Bleu_1",
        "BLEU_2": "Metrics/Bleu_2",
        "BLEU_3": "Metrics/Bleu_3",
        "BLEU_4": "Metrics/Bleu_4",
        "ROUGE_L": "Metrics/ROUGE_L",
    }
    callbacks.append(
        progressLogger(logger,metric_monitor=metric_monitor,log_every_n_steps=1))

    checkpoint_cfg = cfg.get('CHECKPOINT', {})
    if checkpoint_cfg.get('ENABLE_SAFE_CHECKPOINT', False):
        callbacks.append(
            safeCheckpoint(
                dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
                logger=logger,
                every_n_train_steps=checkpoint_cfg.get('EVERY_N_TRAIN_STEPS', 0),
                every_n_epochs=checkpoint_cfg.get('EVERY_N_EPOCHS', 1),
                keep_epoch_checkpoints=checkpoint_cfg.get('KEEP_EPOCH_CHECKPOINTS', True),
                save_on_exception=checkpoint_cfg.get('SAVE_ON_EXCEPTION', True),
                sync_dirpath=checkpoint_cfg.get('SYNC_DIRPATH', None),
                sync_every_n_train_steps=checkpoint_cfg.get('SYNC_EVERY_N_TRAIN_STEPS', 0),
                sync_every_n_epochs=checkpoint_cfg.get('SYNC_EVERY_N_EPOCHS', 0),
                sync_on_exception=checkpoint_cfg.get('SYNC_ON_EXCEPTION', True),
            ))

    # # Save latest checkpoints
    # checkpointParams = {
    #     'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
    #     'filename': "{epoch}",
    #     'monitor': "step",
    #     'mode': "max",
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 8,
    #     'save_last': True,
    #     'save_on_train_epoch_end': True
    # }
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    # # Save checkpoint every n*10 epochs
    # checkpointParams.update({
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS * 10,
    #     'save_top_k': -1,
    #     'save_last': False
    # })
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': None,  #cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
        'save_last': True, #None,
        'save_on_train_epoch_end': False
    }
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    metrics = cfg.METRIC.TYPE
    metric_monitor_map = {
        'TemosMetric': {
            'Metrics/APE_root': {
                'abbr': 'APEroot',
                'mode': 'min'
            },
        },
        'TM2TMetrics': {
            'Metrics/how2sign_DTW_MPJPE_PA_lhand': {
                'abbr': 'how2sign_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/how2sign_DTW_MPJPE_PA_body': {
            #     'abbr': 'how2sign_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # },
            'Metrics/csl_DTW_MPJPE_PA_lhand': {
                'abbr': 'csl_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/csl_DTW_MPJPE_PA_body': {
            #     'abbr': 'csl_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # }
            'Metrics/phoenix_DTW_MPJPE_PA_lhand': {
                'abbr': 'phoenix_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/phoenix_DTW_MPJPE_PA_body': {
            #     'abbr': 'phoenix_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # }
        },
        'M2TMetrics': {
            'Metrics/Bleu_4': {
                'abbr': 'BLEU_4',
                'mode': 'max'
            },
            'Metrics/ROUGE_L': {
                'abbr': 'ROUGE_L',
                'mode': 'max'
            },
        },
        'MRMetrics': {
            'Metrics/how2sign_MPJPE_PA_hand': {
                'abbr': 'how2sign_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/how2sign_MPVPE_PA_all': {
            #     'abbr': 'how2sign_MPVPE_PA_all',
            #     'mode': 'min'
            # },
            'Metrics/csl_MPJPE_PA_hand': {
                'abbr': 'csl_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/csl_MPVPE_PA_all': {
            #     'abbr': 'csl_MPVPE_PA_all',
            #     'mode': 'min'
            # },
            'Metrics/phoenix_MPJPE_PA_hand': {
                'abbr': 'phoenix_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/phoenix_MPVPE_PA_all': {
            #     'abbr': 'phoenix_MPVPE_PA_all',
            #     'mode': 'min'
            # },
        },
        'HUMANACTMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UESTCMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UncondMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            }
        }
    }

    # checkpointParams.update({
    #     'every_n_epochs': None,  #cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 1,
    # })

    for metric in metrics:
        if metric in metric_monitor_map.keys():
            metric_monitors = dict(metric_monitor_map[metric])
            monitor_datasets = checkpoint_cfg.get('MONITOR_DATASETS', None)
            if monitor_datasets:
                monitor_datasets = set(monitor_datasets)
                metric_monitors = {
                    monitor: value
                    for monitor, value in metric_monitors.items()
                    if _metric_allowed_for_datasets(monitor, monitor_datasets)
                }

            # Delete R3 if training VAE
            if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
                del metric_monitors['Metrics/R_precision_top_3']

            for metric_monitor in metric_monitors:
                checkpointParams.update({
                    'filename':
                    metric_monitor_map[metric][metric_monitor]['mode']
                    + "-" +
                    metric_monitor_map[metric][metric_monitor]['abbr']
                    + "{epoch}",
                    'monitor':
                    metric_monitor,
                    'mode':
                    metric_monitor_map[metric][metric_monitor]['mode'],
                })
                callbacks.append(
                    ModelCheckpoint(**checkpointParams))
    return callbacks


def _metric_allowed_for_datasets(monitor, datasets):
    metric_name = monitor.split('/')[-1]
    dataset = metric_name.split('_', 1)[0]
    if dataset in {'how2sign', 'csl', 'phoenix'}:
        return dataset in datasets
    return True

class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class safeCheckpoint(Callback):
    def __init__(self,
                 dirpath,
                 logger=None,
                 every_n_train_steps=0,
                 every_n_epochs=1,
                 keep_epoch_checkpoints=True,
                 save_on_exception=True,
                 sync_dirpath=None,
                 sync_every_n_train_steps=0,
                 sync_every_n_epochs=0,
                 sync_on_exception=True):
        self.dirpath = dirpath
        self.logger = logger
        self.every_n_train_steps = int(every_n_train_steps or 0)
        self.every_n_epochs = int(every_n_epochs or 0)
        self.keep_epoch_checkpoints = keep_epoch_checkpoints
        self.save_on_exception = save_on_exception
        self.sync_dirpath = sync_dirpath
        self.sync_every_n_train_steps = int(sync_every_n_train_steps or 0)
        self.sync_every_n_epochs = int(sync_every_n_epochs or 0)
        self.sync_on_exception = sync_on_exception
        self._last_saved_step = -1
        self._last_synced_step = -1

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def _save_checkpoint(self, trainer, filename):
        if not trainer.is_global_zero:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        target_path = os.path.join(self.dirpath, filename)
        tmp_path = target_path + ".tmp"
        trainer.save_checkpoint(tmp_path)
        os.replace(tmp_path, target_path)
        self._log(f"Safe checkpoint saved to {target_path}")

    def _sync_checkpoint(self, filename):
        if not self.sync_dirpath:
            return
        source_path = os.path.join(self.dirpath, filename)
        if not os.path.exists(source_path):
            self._log(f"Skip checkpoint sync because source is missing: {source_path}")
            return
        os.makedirs(self.sync_dirpath, exist_ok=True)
        target_path = os.path.join(self.sync_dirpath, filename)
        tmp_path = target_path + ".tmp"

        # Google Drive allows duplicate names and its Colab FUSE mount can turn
        # os.replace into another same-name object. Copy a complete temp file
        # first, then delete old same-name targets so Drive keeps one canonical
        # last.ckpt/interrupted.ckpt without losing the previous checkpoint if
        # the upload is interrupted mid-copy.
        self._remove_sync_duplicates(filename + ".tmp")

        shutil.copy2(source_path, tmp_path)
        self._remove_sync_duplicates(filename)
        os.replace(tmp_path, target_path)
        self._log(f"Safe checkpoint synced to {target_path}")

    def _remove_sync_duplicates(self, filename, max_attempts=20):
        if not self.sync_dirpath:
            return
        target_path = os.path.join(self.sync_dirpath, filename)
        removed = 0
        for _ in range(max_attempts):
            if not os.path.exists(target_path):
                break
            try:
                os.remove(target_path)
                removed += 1
            except FileNotFoundError:
                break
            except OSError as exc:
                self._log(f"Could not remove old synced checkpoint {target_path}: {exc}")
                break
        if removed:
            self._log(f"Removed {removed} old synced checkpoint file(s) named {filename}")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs, batch, batch_idx):
        if self.every_n_train_steps <= 0:
            return
        step = int(trainer.global_step)
        if step > 0 and step % self.every_n_train_steps == 0 and step != self._last_saved_step:
            self._save_checkpoint(trainer, "last.ckpt")
            self._last_saved_step = step
        if self.sync_every_n_train_steps > 0 and step > 0 and step % self.sync_every_n_train_steps == 0 and step != self._last_synced_step:
            self._sync_checkpoint("last.ckpt")
            self._last_synced_step = step

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.every_n_epochs <= 0:
            return
        epoch_number = int(trainer.current_epoch) + 1
        if epoch_number % self.every_n_epochs == 0:
            self._save_checkpoint(trainer, "last.ckpt")
            sync_epoch = self.sync_every_n_epochs > 0 and epoch_number % self.sync_every_n_epochs == 0
            if sync_epoch:
                self._sync_checkpoint("last.ckpt")
            if self.keep_epoch_checkpoints:
                epoch_filename = f"epoch-{epoch_number:03d}.ckpt"
                self._save_checkpoint(trainer, epoch_filename)
                if sync_epoch:
                    self._sync_checkpoint(epoch_filename)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception):
        if not self.save_on_exception:
            return
        self._save_checkpoint(trainer, "interrupted.ckpt")
        self._save_checkpoint(trainer, "last.ckpt")
        if self.sync_on_exception:
            self._sync_checkpoint("interrupted.ckpt")
            self._sync_checkpoint("last.ckpt")

class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)
