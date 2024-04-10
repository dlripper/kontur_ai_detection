import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def loss_calc(loss_function, outputs, labels):
    if isinstance(loss_function, torch.nn.modules.loss.CrossEntropyLoss):
         return loss_function(outputs, labels) 
    elif isinstance(loss_function, torch.nn.modules.loss.BCELoss):
        return loss_function(outputs.reshape(-1).sigmoid(), labels.to(torch.float32))
    else:
        raise NotImplementedError("This function is not implemented yet")


def predicted_calc(loss_function, outputs):
    if isinstance(loss_function, torch.nn.modules.loss.CrossEntropyLoss):
        _, predicted = torch.max(outputs.data, 1)
        return predicted
    elif isinstance(loss_function, torch.nn.modules.loss.BCELoss):
        return (outputs.reshape(-1).data.sigmoid() >= 0.5).to(torch.int32)
    else:
        raise NotImplementedError("This function is not implemented yet")


def train_model(model, optimizer, loss_function, train_dataloader, test_dataloader, num_epochs, wandb_specs=None):
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
