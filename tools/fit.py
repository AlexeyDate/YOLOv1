import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def evaluate(model, criterion, val_dataloader, device):
    val_loss = 0
    model.eval()
    for batch in tqdm(val_dataloader, desc=f'Evaluation', leave=False):
        images, targets = batch['image'], batch['target']
        images, targets = images.to(device), targets.to(device)

        with torch.no_grad():
            predictions = model(images)
            loss = criterion(predictions, targets)
            val_loss += loss.item()

    return val_loss / len(val_dataloader)


def fit(model, optimizer, scheduler, criterion, epochs, train_dataloader, val_dataloader, device='cpu'):
    """
    training model with drawing graph of the loss curve

    param: model - model to fitting
    param: optimizer - optimizer loss function
    param: scheduler - optimizer scheduler
    param: criterion - loss function
    param: epochs - number of epochs
    param: train_dataloader - dataloader with training split of dataset
    param: val_dataloader - dataloader with testing split of dataset
    param: device - device of model (default = cpu)
    """

    train_loss_log = []
    val_loss_log = []
    fig = plt.figure(figsize=(11, 7))
    fig_number = fig.number

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training, epoch {epoch}", leave=False):
            images, targets = batch['image'], batch['target']
            images, targets = images.to(device), targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        train_loss /= len(train_dataloader)
        train_loss_log.append(train_loss)
        val_loss = evaluate(model, criterion, val_dataloader, device)
        val_loss_log.append(val_loss)

        if not plt.fignum_exists(num=fig_number):
            fig = plt.figure(figsize=(11, 7))
            fig_number = fig.number

        print(f"epoch: {epoch}")
        print(f"train loss: {train_loss}")
        print(f"val loss: {val_loss}")
        line_train, = plt.plot(list(range(0, epoch + 1)), train_loss_log, color='blue')
        line_val, = plt.plot(list(range(0, epoch + 1)), val_loss_log, color='orange')
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Train steps")
        plt.legend((line_train, line_val), ['train loss', 'validation loss'])
        plt.draw()
        plt.pause(0.001)
        fig.savefig('loss.png', bbox_inches='tight')

