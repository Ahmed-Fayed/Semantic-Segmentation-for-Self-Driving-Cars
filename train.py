import torch.nn as nn
import torch.onnx
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import random
import numpy as np
import time
import gc
from config import *
from Utils.utils import epoch_time, plot_results, calculate_accuracy

from Utils.models import UNet
from Utils.dataset import train_iterator, val_iterator

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


def train(train_iterator, model, criterion, optimizer, scaler, device):
    train_size = len(train_iterator)
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (data, targets) in train_iterator:
        # compute predictions and loss
        data = data.to(device)
        targets = targets.to(device)
        targets = targets.type(torch.long)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = criterion(predictions, targets)

        # Backprobagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accuracy = calculate_accuracy(predictions, targets)

        epoch_loss += loss.item()
        epoch_acc += accuracy

    epoch_loss /= train_size
    epoch_acc /= train_size

    return epoch_loss, epoch_acc


def evaluate(val_iterator, model, criterion, device):
    val_size = len(val_iterator)
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (data, targets) in val_iterator:
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.type(torch.long)

            predictions = model(data)

            loss = criterion(predictions, targets)

            accuracy = calculate_accuracy(predictions, targets)

            epoch_loss += loss.item()
            epoch_acc += accuracy

    epoch_loss /= val_size
    epoch_acc /= val_size

    return epoch_loss, epoch_acc


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


if __name__ == "__main__":

    print(f'torch version: {torch.__version__}')
    gc.collect()
    torch.cuda.empty_cache()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.set_experiment("Aerial-Imaginary-Segmentation-1")
    print(f"experiments: '{mlflow.search_experiments()}'")

    print("training started!")
    with mlflow.start_run():
        # Model Initialization
        model = UNet(num_classes=NUM_CLASSES)
        # model = TransUNet(n_channels=3, n_classes=NUM_CLASSES)

        try:
            if os.path.exists(model_name):
                model.load_state_dict(torch.load(model_name))
                print("found weights, loaded successfully!")
        except Exception as e:
            print(f"couldn't find pre-trained weights, {e}")

        print(f"UNet Summery: {model}")

        # loss function
        criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss(gamma=3 / 4)

        # Optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

        # Grad Scaler
        scaler = torch.cuda.amp.GradScaler()

        print(f"device: {DEVICE}")

        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)

        MAX_LRS = [p['lr'] for p in optimizer.param_groups]
        STEPS_PER_EPOCH = len(train_iterator)
        TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
        # scheduler = lr_scheduler.OneCycleLR(optimizer,
        #                                     max_lr=MAX_LRS,
        #                                     total_steps=TOTAL_STEPS)

        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        best_valid_loss = float('inf')

        params = {"epochs": EPOCHS, "learning_rate": learning_rate, "criterion": criterion, "optimizer": optimizer,
                  "scheduler": scheduler, "scaler": scaler, "random_state": SEED}

        mlflow.log_params(params)

        plot_losses = {"train_loss": [], "val_loss": []}
        plot_accuracy = {"train_acc": [], "val_acc": []}
        scheduler_counter = 0

        for epoch in tqdm(range(EPOCHS)):

            start_time = time.monotonic()

            train_loss, train_acc = train(train_iterator, model, criterion, optimizer, scaler, DEVICE)
            valid_loss, valid_acc = evaluate(val_iterator, model, criterion, DEVICE)

            plot_losses["train_loss"].append(train_loss)
            plot_losses["val_loss"].append(valid_loss)
            plot_accuracy["train_acc"].append(train_acc)
            plot_accuracy["val_acc"].append(valid_acc)

            scheduler_counter += 1
            if valid_loss < best_valid_loss:
                scheduler_counter = 0
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_name)

            if scheduler_counter > 5:
                scheduler.step()
                print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
                scheduler_counter = 0

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:6.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:6.2f}%')

        # plotting and saving results
        plot_results(plot_losses["train_loss"], plot_losses["val_loss"], "Loss", True, "Loss")
        plot_results(plot_accuracy["train_acc"], plot_accuracy["val_acc"], "Loss", True, "Accuracy")

        mlflow.log_metric("Train Loss", round(train_loss, 3))
        mlflow.log_metric("Train Acc", round(train_acc.item() * 100, 2))
        mlflow.log_metric("Valid Loss", round(valid_loss, 3))
        mlflow.log_metric("Valid Acc", round(valid_acc.item() * 100, 2))
        mlflow.pytorch.log_model(model, artifact_path="models")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

    print(f"experiments: '{mlflow.search_experiments()}'")

    client = MlflowClient("http://127.0.0.1:5000")

    try:
        print(f" Registered Models: {client.search_registered_models()}")
    except MlflowException:
        print("It's not possible to access the model registry :(")

    run_id = client.search_runs(experiment_ids='1')[0].info.run_id
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/models",
        name='UNet-Segmentation'
    )

    try:
        print(f" Registered Models: {client.search_registered_models()}")
    except MlflowException:
        print("It's not possible to access the model registry :(")
