import torch
import torch.nn as nn
import wandb
from tqdm import tqdm


def train_model(model, optimizer, loss, train_dataloader, test_dataloader, num_epochs, wandb_specs=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
        }, "data/buffer_png.pth")

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
            outputs = model(inputs).reshape(-1)

            loss = criterion(outputs.sigmoid(), labels.to(torch.float32)) 
            
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            total_len += inputs.size(0)
            predicted = (outputs.data.sigmoid() >= 0.5).to(torch.int32)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if iter_num % 50 == 0:
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, labels, _ in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs).reshape(-1)
                        loss = criterion(outputs.sigmoid(), labels.to(torch.float32)) 
                        val_loss += loss.item() * inputs.size(0)
                        predicted = (outputs.data.sigmoid() >= 0.5).to(torch.int32)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                            val_loss = val_loss / len(test_dataloader.dataset)
                val_acc = val_correct / val_total
            
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(torch.load("data/sota_3.pth"), "data/sota_4.pth")
                    torch.save(torch.load("data/sota_2.pth"), "data/sota_3.pth")
                    torch.save(torch.load("data/sota_1.pth"), "data/sota_2.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, "data/sota_1.pth")
   
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
                outputs = model(inputs).reshape(-1)
                loss = criterion(outputs.sigmoid(), labels.to(torch.float32)) 
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs.data.sigmoid() >= 0.5).to(torch.int32)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(test_dataloader.dataset)
        val_acc = val_correct / val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(torch.load("data/sota_3.pth"), "data/sota_4.pth")
            torch.save(torch.load("data/sota_2.pth"), "data/sota_3.pth")
            torch.save(torch.load("data/sota_1.pth"), "data/sota_2.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "data/sota_1.pth")
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if wandb_specs is not None:
            wandb.log({"Train Loss": train_loss, "Train Acc": train_acc, "Val Loss": val_loss, "Val Acc": val_acc})
