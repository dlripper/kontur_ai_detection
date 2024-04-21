import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from typing import Optional, Union, Dict
from tqdm import tqdm


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def loss_calc(
    loss_function: nn.Module, 
    outputs: torch.Tensor, 
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Calculate loss based on the given loss function.

    Args:
        loss_function: The loss function to use for calculation (e.g., CrossEntropyLoss, BCELoss).
        outputs: The model's output predictions.
        labels: The ground truth labels.

    Returns:
        The computed loss as a tensor.

    Raises:
        NotImplementedError: If the loss function is not implemented.
    """
    if isinstance(loss_function, torch.nn.modules.loss.CrossEntropyLoss):
         return loss_function(outputs, labels) 
    elif isinstance(loss_function, torch.nn.modules.loss.BCELoss):
        return loss_function(outputs.reshape(-1).sigmoid(), labels.to(torch.float32))
    else:
        raise NotImplementedError("This function is not implemented yet")


def predicted_calc(
    loss_function: nn.Module, 
    outputs: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the predicted class or value based on the given loss function.

    Args:
        loss_function: The loss function to determine the prediction logic.
        outputs: The model's output predictions.

    Returns:
        The predicted classes or binary outputs.

    Raises:
        NotImplementedError: If the prediction logic is not implemented for the given loss function.
    """
    if isinstance(loss_function, torch.nn.modules.loss.CrossEntropyLoss):
        _, predicted = torch.max(outputs.data, 1)
        return predicted
    elif isinstance(loss_function, torch.nn.modules.loss.BCELoss):
        return (outputs.reshape(-1).data.sigmoid() >= 0.5).to(torch.int32)
    else:
        raise NotImplementedError("This function is not implemented yet")


WandbSpecsType = Optional[Dict[str, Union[str, Dict[str, Union[str, int]]]]]


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    wandb_specs: WandbSpecsType = None,
) -> None:
    """
    Train the model with the given optimizer, loss function, and data loaders.

    Args:
        model: The model to train.
        optimizer: The optimizer used for updating weights.
        loss_function: The loss function to use during training.
        train_dataloader: The data loader for the training set.
        test_dataloader: The data loader for the validation set.
        num_epochs: The number of epochs to train for.
        wandb_specs: Optional configuration for logging with Weights & Biases.

    Returns:
        None

    Raises:
        NotImplementedError: If the can't calc the provided loss_function
    """
    model.to(device)
    if wandb_specs is not None:
        wandb.login()
        wandb.init(
            project=wandb_specs["project"], 
            name=wandb_specs["name"], 
            config=wandb_specs["config"]
        )

    best_val_loss = float('inf')

    iter_num = 0
    for epoch in range(num_epochs):
        if epoch == 0:
            #create .pth objects
            for i in range(4):
                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }, f"weights/sota_{i}.pth")
                                
        torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
        }, "weights/buffer_png.pth")

        #####################
        #####TRAIN PART######
        #####################
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        total_len = 0
        for inputs, labels, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            iter_num += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_calc(loss_function, outputs, labels)
                 
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            total_len += inputs.size(0)
            predicted = predicted_calc(loss_function, outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if iter_num % 50 == 0:
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, labels, _ in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = loss_calc(loss_function, outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        predicted = predicted_calc(loss_function, outputs)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(test_dataloader.dataset)
                val_acc = val_correct / val_total
            
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(torch.load("weights/sota_3.pth"), "weights/sota_4.pth")
                    torch.save(torch.load("weights/sota_2.pth"), "weights/sota_3.pth")
                    torch.save(torch.load("weights/sota_1.pth"), "weights/sota_2.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, "weights/sota_1.pth")
   
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = correct / total

        #####################
        #####EVAL PART#######
        #####################
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels, _ in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_calc(loss_function, outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = predicted_calc(loss_function, outputs)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(test_dataloader.dataset)
        val_acc = val_correct / val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(torch.load("weights/sota_3.pth"), "weights/sota_4.pth")
            torch.save(torch.load("weights/sota_2.pth"), "weights/sota_3.pth")
            torch.save(torch.load("weights/sota_1.pth"), "weights/sota_2.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "weights/sota_1.pth")
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if wandb_specs is not None:
            wandb.log({"Train Loss": train_loss, "Train Acc": train_acc, "Val Loss": val_loss, "Val Acc": val_acc})
