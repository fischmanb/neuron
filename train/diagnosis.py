import os
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader

from train import trainroot
from train.datasets.diagnosis import DiagClip
from train.metrics.wandb import WandbLogger, init_wandb
from train.networks.mfcc import ModernM5
from train.util.storage import save_model


def train_loop(config, device):
    run_name = f"{config['run_name']}-{str(config.get('ax_counter')).zfill(2)}" if config.get("ax_counter") \
        else config['run_name']

    train_ds = DiagClip(split="train")
    val_ds = DiagClip(split="val")

    print(f"{run_name} Dataset {train_ds.__class__.__name__} Train {len(train_ds)} Val {len(val_ds)}")

    dataloaders = dict({
        'train': DataLoader(train_ds, batch_size=config.get('batch_size'),
                            num_workers=config.get('num_workers'), pin_memory=False,
                            drop_last=True, shuffle=True),
        'val': DataLoader(val_ds, batch_size=config.get('batch_size'),
                          num_workers=config.get('num_workers'), pin_memory=False,
                          drop_last=True, shuffle=True)
    })

    model = ModernM5(num_classes=config['num_classes']).to(device)
    # print("Setting up wandb model watch")
    # wandb.watch(model)
    model = model.to(device)

    if (torch.cuda.device_count() > 1) and device != 'cpu' and device != 'mps':
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        print(f"USING GPU count: {torch.cuda.device_count()}")
        
    xx = config['pos_multiplier']

    lossweights = torch.tensor(config['num_classes'] * [xx])

    loss_function = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=lossweights).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.get('learning_rate'), amsgrad=True, weight_decay=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_lr"], gamma=config["step_gamma"])
    # UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.

    sigmoid = nn.Sigmoid()

    config["train size"] = str(len(train_ds))  # str() so chart isn't rendered in wandb
    config["val size"] = str(len(val_ds))
    config['model_name'] = model.name if model.name else f"Class {model.split('(')[0]}"
    config['ds_name'] = str(train_ds.__class__)
    init_wandb(wandb, config)

    assert os.path.exists(trainroot / "configs" / config["configfile"]), "Could not find config file"
    try:
        wandb.save(trainroot / "configs" / config["configfile"])
    except Exception as err:
        print(f"Error saving config file to wandb: {err}")

    wb = WandbLogger(wandb, run_name=run_name)

    print('\nTraining Parameters:\n')
    for k, v in config.items():
        print(k + '\t' + str(v))

    for phase in ['train', 'val']:
        print(f"{phase} num batches {len(dataloaders[phase])}")

    best_val_acc = 0.0

    for epoch in range(1, config["num_epochs"] + 1):

        running_y_true = np.empty((0, config['num_classes']))
        running_guesses = np.empty((0, config['num_classes']))
        running_preds = np.empty((0, config['num_classes']))

        for phase in ['train', 'val']:
            if phase == 'train':
                print("Train Phase")
                model.train()
                torch.set_grad_enabled(True)
            elif phase == 'val':
                print("Validation Phase")
                model.eval()
                torch.set_grad_enabled(False)

            running_loss = 0.0
            running_corrects = 0
            for step, batch in enumerate(dataloaders[phase], 1):
                vecs = batch['vecs'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad()
                preds = model(vecs)
                loss = loss_function(preds, labels.float())  # TODO fix this in the dataset or the train loop or the network later
                running_loss += loss
                
                guesses = sigmoid(preds) > config['multilabel_thresh']
                batch_corrects = torch.sum(guesses == labels)
                running_corrects += batch_corrects

                if step % 150 == 0 or step == 1:
                    print(f"{len(dataloaders[phase]) - step} avg loss {running_loss / step} sample step loss {loss.item()} batch corrects {batch_corrects} avg acc {batch_corrects / (config['batch_size'] * config['num_classes'])}")

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if phase == 'val':
                    # # append values being collected for confusion matrix
                    _labels = labels.detach().cpu().numpy()
                    running_y_true = np.concatenate((running_y_true, _labels), axis=0)

                    _guesses = guesses.int().detach().cpu().numpy()
                    running_guesses = np.concatenate((running_guesses, _guesses), axis=0)

                    _preds = sigmoid(preds).detach().cpu().numpy()
                    running_preds = np.concatenate((running_preds, _preds), axis=0)

            # EPOCH END
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / (config['num_classes'] * len(dataloaders[phase]) * config['batch_size'])
            print(f"{epoch} / {phase} - loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")
            wb.eplog(epoch, "LR", scheduler.get_last_lr()[-1])
            wb.eplog(epoch, f"{phase}_loss", epoch_loss)
            wb.eplog(epoch, f"{phase}_acc", epoch_acc)
            
            if phase == 'train':
                scheduler.step()
            elif phase == 'val':
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    print(f"New best val epoch acc - epoch: {epoch} best_acc: {best_val_acc:.4f}")

            if epoch >= 70:
                save_model(trainroot / 'weights', 'abc,123', 'multilabel',
                           run_name, epoch, model, optimizer, scheduler)

        cmat = multilabel_confusion_matrix(running_y_true, running_guesses)  # normalize="true"
        print(cmat)

        prec_report = classification_report(running_y_true, running_guesses, zero_division=np.nan)
        print(prec_report)
        for klass_idx in range(config['num_classes']):
            cmp = ConfusionMatrixDisplay(confusion_matrix=cmat[klass_idx])
            plt.autoscale()
            fig, ax = plt.subplots(figsize=(9, 9))
            cmp.plot(ax=ax, xticks_rotation="vertical")
            fig.suptitle(f"Epoch {epoch} Val - {train_ds.idx_to_label[klass_idx]}")
            fig.tight_layout(pad=1)
            wb.log({f"cmat/val {train_ds.idx_to_label[klass_idx]}": fig})
            plt.close()

        for klass_idx in range(config['num_classes']):
            xlabels = running_y_true[:, klass_idx]
            xpreds = np.stack([1 - running_preds[:, klass_idx], running_preds[:, klass_idx]], axis=-1)
            xlegend = [f'{train_ds.idx_to_label[klass_idx]} 0', f'{train_ds.idx_to_label[klass_idx]} 1']
            wandb.log({f"roc/{train_ds.idx_to_label[klass_idx]}": wandb.plot.roc_curve(xlabels, xpreds,
                                                                                          xlegend, title=
                                                                                          train_ds.idx_to_label[
                                                                                              klass_idx])})

        for klass_idx in range(config['num_classes']):
            klass_preds = running_preds[:, klass_idx]
            klass_labels = running_y_true[:, klass_idx]
            gt_true = klass_labels == 1
            gt_false = klass_labels == 0
            pred_true = klass_preds > 0.5
            pred_false = klass_preds <= 0.5
            tp = sum(np.logical_and(gt_true, pred_true))
            tn = sum(np.logical_and(gt_true, pred_false))
            fp = sum(np.logical_and(gt_false, pred_false))
            fn = sum(np.logical_and(gt_false, pred_true))
            print(f"{train_ds.idx_to_label[klass_idx]}")
            print(f"   total True {sum(gt_true)} TP {tp} TN {tn}")
            print(f"   total False {sum(gt_false)} FP {fp} FN {fn}")
            sens = tp / (tp + fn)  # TODO show more meaningful message when tp + fn == 0 (div by zero)
            spec = tn / (tn + fp)
            print(f"   Sensitivity {sens} -- Specificity {spec}")
            wb.eplog(epoch, f"{phase}_TP", tp)
            wb.eplog(epoch, f"{phase}_TN", tn)
            wb.eplog(epoch, f"{phase}_FP", fp)
            wb.eplog(epoch, f"{phase}_FN", fn)
            wb.eplog(epoch, f"{phase}_Sensitivity", sens)
            wb.eplog(epoch, f"{phase}_Specificity", spec)

        if epoch == config["num_epochs"] and config["disable_ax"]:
            save_model(trainroot / 'weights', ','.join(['list', 'of', 'classlabels']),  # train_ds.classlabels
                       config['task_name'], run_name, epoch, model, optimizer, scheduler)
    # Standard error of the metric's mean, 0.0 for noiseless measurements.
    wandb.finish()
    # return {"best_roc_auc": (best_roc_auc, 0.0)}  # early stopping not triggered, num_epochs completed


def main():
    manualconf = {
        "task_name": "MFCC",
        "num_classes": 9,
        "disable_ax": True,
        "ax_total_trials": 0,
        "multilabel_thresh": 0.6,
        "configfile": "localonly.ini",
        "num_workers": 10,
        "local_debug": False,
        "tracking_project": "features",
        "num_epochs": 10,
        "batch_size": 1,
        "fine_tune_epochs": 0,
        "fine_tune_lr_factor": 0.5,
        "learning_rate": 1e-04,  #  1e-5 to 5e-5,
        "step_lr": 23,
        "step_gamma": 0.8,
        "run_name": datetime.fromtimestamp(time()).strftime('%d_%B_%H%M%p'),  # never change this
        "grad_sal": False,
        "pos_multiplier": 1,  # positive class for loss weight
    }

    if torch.backends.mps.is_available():
        device = 'cpu'
        manualconf['num_epochs'] = 2
        manualconf['num_workers'] = 0
        manualconf['batch_size'] = 8
        manualconf['local_debug'] = True
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        manualconf['num_epochs'] = 1
        manualconf['num_workers'] = 1
        manualconf['batch_size'] = 8
        manualconf['local_debug'] = True

    print(f"Device {device}")
    train_loop(manualconf, device)


if __name__ == '__main__':
    main()
