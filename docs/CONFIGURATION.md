# Configuration System

This project uses a strict, hierarchical YAML configuration system to manage experiments. 

### **1. CRITICAL RULE**
**No hardcoded experiment hyperparameters are allowed inside Python code.** 
Every hyperparameter, dimension, or architectural flag must be injected via these configuration files. This ensures 100% reproducibility and prevents untracked local modifications.

---

### **2. Configuration Hierarchy**
The configuration is divided into three levels:

* `configs/base.yaml`
* `configs/models/`
* `configs/recipes/`

#### **Base Configuration (`base.yaml`)**
Contains universal project settings that rarely change across different runs.
* **Belongs here:** Dataset paths (CIFAR-100), global seed, logging output directories, image sizes (e.g., 32x32), dataloader workers, and pin memory settings.

#### **Model Configuration (`models/*.yaml`)**
Contains architecture-specific hyperparameters.
* **Belongs here:** Architecture names (e.g., `resnet18`, `vit_tiny`), embedding dimensions, depth, patch sizes, and model-specific structural flags (like `modify_stem: true` for ResNet CIFAR adaptation).

#### **Recipe Configuration (`recipes/*.yaml`)**
Contains the training intervention (the "Recipe").
* **Belongs here:** Epochs, batch size, optimizer type and parameters (lr, weight decay, momentum), learning rate schedules (warmup, minimum lr), and augmentation logic (RandAugment, Mixup, Cutmix probabilities, label smoothing).

---

### **3. Configuration Merging**
Configurations are dynamically merged at runtime using the `merge_configs()` utility in `src/utils/config.py`. 
The merging process is recursive. The base configuration is loaded first, followed by the model config, and finally the recipe config. In case of conflicting keys, the later configs override the earlier ones (e.g., Recipe > Model > Base).

---

### **4. Example Experiment Launch**
To run an experiment, you specify the configuration components via command-line arguments. The script will load and merge them automatically:

```bash
python scripts/train.py --model configs/models/resnet18.yaml --recipe configs/recipes/recipe_a.yaml
```
*(Note: `base.yaml` is loaded automatically by the training script).*
