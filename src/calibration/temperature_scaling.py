import torch
import torch.nn as nn
import torch.optim as optim

def fit_temperature_scaling(model: nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Fits a single temperature scaling parameter using L-BFGS on the clean validation set.
    """
    model.eval()
    nll_criterion = nn.CrossEntropyLoss().to(device)
    
    # Collect all logits and labels
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits_list.append(model(inputs))
            labels_list.append(labels)
            
    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)
    
    # Initialize temperature parameter
    temperature = nn.Parameter(torch.ones(1, device=device))
    
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval_fn():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_fn)
    
    return temperature.item()
