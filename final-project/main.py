import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from PIL import Image
import os
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math


class ImagePreprocessor:
    """Handle image transformations and normalization"""
    
    def __init__(self, img_size=224, augment=True):
        self.img_size = img_size
        self.augment = augment
    
    def get_train_transforms(self):
        """Returns training transforms with augmentation"""
        if self.augment:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def get_val_transforms(self):
        """Returns validation/test transforms without augmentation"""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class DatasetSplitter:
    """Manage train/validation/test splits"""
    
    def __init__(self, dataset_path, val_split=0.2):
        self.dataset_path = dataset_path
        self.val_split = val_split
    
    def load_datasets(self):
        """Load and split datasets"""
        train_dir = os.path.join(self.dataset_path, 'train')
        val_dir = os.path.join(self.dataset_path, 'val')
        test_dir = os.path.join(self.dataset_path, 'test')
        
        if os.path.exists(val_dir):
            return train_dir, val_dir, test_dir
        else:
            train_dataset = datasets.ImageFolder(train_dir)
            train_size = len(train_dataset)
            val_size = int(self.val_split * train_size)
            train_size = train_size - val_size
            
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            return train_dataset, val_dataset, test_dir
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        train_dir = os.path.join(self.dataset_path, 'train')
        class_counts = {}
        
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len(os.listdir(class_path))
        
        total_samples = sum(class_counts.values())
        class_weights = {}
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (len(class_counts) * count)
        
        return class_weights


class PneumoniaDataLoader:
    """Handle dataset loading, preprocessing, and batching"""
    
    def __init__(self, dataset_path, batch_size=32, img_size=224, val_split=0.2):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_split = val_split
        self.preprocessor = ImagePreprocessor(img_size)
        self.splitter = DatasetSplitter(dataset_path, val_split)
    
    def create_data_loaders(self):
        """Create train, validation, and test data loaders"""
        train_transforms = self.preprocessor.get_train_transforms()
        val_transforms = self.preprocessor.get_val_transforms()
        
        train_dir = os.path.join(self.dataset_path, 'train')
        val_dir = os.path.join(self.dataset_path, 'val')
        test_dir = os.path.join(self.dataset_path, 'test')
        
        if os.path.exists(val_dir):
            train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
            val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        else:
            full_train_dataset = datasets.ImageFolder(train_dir)
            train_size = len(full_train_dataset)
            val_size = int(self.val_split * train_size)
            train_size = train_size - val_size
            
            train_indices, val_indices = random_split(range(len(full_train_dataset)), [train_size, val_size])
            
            train_dataset = torch.utils.data.Subset(
                datasets.ImageFolder(train_dir, transform=train_transforms), 
                train_indices.indices
            )
            val_dataset = torch.utils.data.Subset(
                datasets.ImageFolder(train_dir, transform=val_transforms), 
                val_indices.indices
            )
        
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    
    def get_dataset_stats(self):
        """Get dataset statistics"""
        stats = {}
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.dataset_path, split)
            if os.path.exists(split_dir):
                stats[split] = {}
                for class_name in os.listdir(split_dir):
                    class_path = os.path.join(split_dir, class_name)
                    if os.path.isdir(class_path):
                        stats[split][class_name] = len(os.listdir(class_path))
        return stats


class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings for ViT"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """Feed-forward network for transformer"""
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model implementation"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=2, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification head
        x = self.norm(x)
        x = x[:, 0]  # Take cls token
        x = self.head(x)
        
        return x
    
    def get_num_params(self):
        """Return number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNModel(nn.Module):
    """Convolutional Neural Network implementation"""
    
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_num_params(self):
        """Return number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MetricsCalculator:
    """Compute accuracy, precision, recall, F1-score"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions, targets):
        """Update metrics with new predictions"""
        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        self.predictions.extend(pred_classes)
        self.targets.extend(targets_np)
    
    def compute(self):
        """Compute final metrics"""
        if len(self.predictions) == 0:
            return {}
            
        accuracy = accuracy_score(self.targets, self.predictions)
        precision = precision_score(self.targets, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(self.targets, self.predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.targets, self.predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_confusion_matrix(self, predictions, targets, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()
    
    def plot_roc_curve(self, predictions, targets):
        """Plot ROC curve"""
        if len(np.unique(targets)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(targets, predictions)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            return plt.gcf()
        return None


class TrainingVisualizer:
    """Plot training curves and results"""
    
    def __init__(self, save_dir="./plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(train_accs, label='Training Accuracy', color='blue')
        ax2.plot(val_accs, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        return fig
    
    def plot_model_comparison(self, cnn_metrics, vit_metrics):
        """Compare CNN vs ViT performance"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        cnn_values = [cnn_metrics[m] for m in metrics]
        vit_values = [vit_metrics[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, cnn_values, width, label='CNN', alpha=0.8)
        ax.bar(x + width/2, vit_values, width, label='ViT', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('CNN vs ViT Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (cnn_val, vit_val) in enumerate(zip(cnn_values, vit_values)):
            ax.text(i - width/2, cnn_val + 0.01, f'{cnn_val:.3f}', ha='center')
            ax.text(i + width/2, vit_val + 0.01, f'{vit_val:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'model_comparison.png'))
        return fig
    
    def plot_training_time_comparison(self, cnn_time, vit_time):
        """Compare training times"""
        models = ['CNN', 'ViT']
        times = [cnn_time, vit_time]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(models, times, color=['skyblue', 'lightcoral'], alpha=0.8)
        
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + time_val*0.01,
                   f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_time_comparison.png'))
        return fig


class ModelTrainer:
    """Handle training loop, validation, and checkpointing"""
    
    def __init__(self, model, device, criterion, optimizer, scheduler=None, 
                 save_dir="./checkpoints"):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.metrics = MetricsCalculator()
        
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        self.metrics.reset()
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.metrics.update(output, target)
        
        avg_loss = total_loss / len(train_loader)
        metrics = self.metrics.compute()
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                self.metrics.update(output, target)
        
        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics.compute()
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):
        """Full training loop with early stopping"""
        best_val_acc = 0
        patience_counter = 0
        training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_acc'].append(train_metrics['accuracy'])
            training_history['val_acc'].append(val_metrics['accuracy'])
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_metrics["accuracy"]:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
        
        return training_history
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch'], checkpoint['metrics']


class ModelEvaluator:
    """Test set evaluation and performance analysis"""
    
    def __init__(self, model, device, criterion):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.metrics = MetricsCalculator()
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                self.metrics.update(output, target)
        
        avg_loss = total_loss / len(test_loader)
        metrics = self.metrics.compute()
        
        return avg_loss, metrics
    
    def detailed_analysis(self, test_loader, class_names):
        """Perform detailed analysis with visualizations"""
        predictions, targets, probs = self.get_predictions(test_loader)
        
        # Confusion matrix
        cm_fig = self.metrics.plot_confusion_matrix(predictions, targets, class_names)
        
        # ROC curve for binary classification
        roc_fig = None
        if len(class_names) == 2:
            roc_fig = self.metrics.plot_roc_curve(probs[:, 1], targets)
        
        return cm_fig, roc_fig
    
    def get_predictions(self, test_loader):
        """Get all predictions and targets"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probs)


class ResultsLogger:
    """Save metrics and model artifacts"""
    
    def __init__(self, save_dir="./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def log_training_results(self, model_name, training_history):
        """Log training history"""
        df = pd.DataFrame(training_history)
        df.to_csv(os.path.join(self.save_dir, f'{model_name}_training_history.csv'), index=False)
        
        # Save as JSON as well
        import json
        with open(os.path.join(self.save_dir, f'{model_name}_training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
    
    def log_evaluation_results(self, model_name, metrics):
        """Log evaluation metrics"""
        import json
        with open(os.path.join(self.save_dir, f'{model_name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create a summary text file
        with open(os.path.join(self.save_dir, f'{model_name}_summary.txt'), 'w') as f:
            f.write(f'{model_name} Evaluation Results\n')
            f.write('=' * 30 + '\n')
            for metric, value in metrics.items():
                f.write(f'{metric.capitalize()}: {value:.4f}\n')
    
    def save_model_summary(self, model_name, model, training_time, num_params):
        """Save model summary and statistics"""
        summary = {
            'model_name': model_name,
            'num_parameters': num_params,
            'training_time_seconds': training_time,
            'model_type': type(model).__name__
        }
        
        import json
        with open(os.path.join(self.save_dir, f'{model_name}_model_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


class ModelComparator:
    """Compare CNN vs ViT performance"""
    
    def __init__(self, results_logger, visualizer):
        self.results_logger = results_logger
        self.visualizer = visualizer
    
    def compare_models(self, cnn_results, vit_results):
        """Compare two models comprehensively"""
        # Plot performance comparison
        self.visualizer.plot_model_comparison(
            cnn_results['metrics'], vit_results['metrics']
        )
        
        # Plot training time comparison
        self.visualizer.plot_training_time_comparison(
            cnn_results['training_time'], vit_results['training_time']
        )
        
        # Generate comparison report
        comparison = self.generate_comparison_report(cnn_results, vit_results)
        
        return comparison
    
    def generate_comparison_report(self, cnn_results, vit_results):
        """Generate detailed comparison report"""
        report = {
            'cnn': {
                'metrics': cnn_results['metrics'],
                'training_time': cnn_results['training_time'],
                'num_params': cnn_results['num_params']
            },
            'vit': {
                'metrics': vit_results['metrics'],
                'training_time': vit_results['training_time'],
                'num_params': vit_results['num_params']
            },
            'comparison': {
                'accuracy_diff': vit_results['metrics']['accuracy'] - cnn_results['metrics']['accuracy'],
                'time_ratio': vit_results['training_time'] / cnn_results['training_time'],
                'param_ratio': vit_results['num_params'] / cnn_results['num_params']
            }
        }
        
        # Save comparison report
        import json
        with open(os.path.join(self.results_logger.save_dir, 'model_comparison.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


class ExperimentRunner:
    """Orchestrate training of both models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_logger = ResultsLogger()
        self.visualizer = TrainingVisualizer()
        self.comparator = ModelComparator(self.results_logger, self.visualizer)
    
    def setup_data(self):
        """Setup data loaders"""
        data_loader = PneumoniaDataLoader(
            dataset_path=self.config['dataset_path'],
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            val_split=self.config['val_split']
        )
        
        self.train_loader, self.val_loader, self.test_loader = data_loader.create_data_loaders()
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
        print(f"Dataset loaded successfully!")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
    
    def create_models(self):
        """Create CNN and ViT models"""
        # CNN Model
        self.cnn_model = CNNModel(
            num_classes=2,
            dropout=self.config['cnn_dropout']
        ).to(self.device)
        
        # ViT Model
        self.vit_model = VisionTransformer(
            img_size=self.config['img_size'],
            patch_size=self.config['vit_patch_size'],
            num_classes=2,
            embed_dim=self.config['vit_embed_dim'],
            depth=self.config['vit_depth'],
            num_heads=self.config['vit_num_heads']
        ).to(self.device)
        
        print(f"CNN Parameters: {self.cnn_model.get_num_params():,}")
        print(f"ViT Parameters: {self.vit_model.get_num_params():,}")
    
    def train_cnn(self):
        """Train CNN model"""
        print("\n" + "="*50)
        print("Training CNN Model")
        print("="*50)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        trainer = ModelTrainer(
            model=self.cnn_model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=os.path.join(self.config['save_dir'], 'cnn_checkpoints')
        )
        
        start_time = time.time()
        training_history = trainer.train(
            self.train_loader,
            self.val_loader,
            num_epochs=self.config['num_epochs'],
            early_stopping_patience=self.config['early_stopping_patience']
        )
        training_time = time.time() - start_time
        
        # Log training results
        self.results_logger.log_training_results('cnn', training_history)
        self.results_logger.save_model_summary(
            'cnn', self.cnn_model, training_time, self.cnn_model.get_num_params()
        )
        
        return training_history, training_time
    
    def train_vit(self):
        """Train ViT model"""
        print("\n" + "="*50)
        print("Training ViT Model")
        print("="*50)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.vit_model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        trainer = ModelTrainer(
            model=self.vit_model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=os.path.join(self.config['save_dir'], 'vit_checkpoints')
        )
        
        start_time = time.time()
        training_history = trainer.train(
            self.train_loader,
            self.val_loader,
            num_epochs=self.config['num_epochs'],
            early_stopping_patience=self.config['early_stopping_patience']
        )
        training_time = time.time() - start_time
        
        # Log training results
        self.results_logger.log_training_results('vit', training_history)
        self.results_logger.save_model_summary(
            'vit', self.vit_model, training_time, self.vit_model.get_num_params()
        )
        
        return training_history, training_time
    
    def evaluate_models(self):
        """Evaluate both models on test set"""
        print("\n" + "="*50)
        print("Evaluating Models on Test Set")
        print("="*50)
        
        results = {}
        
        # Evaluate CNN
        print("\nEvaluating CNN...")
        cnn_evaluator = ModelEvaluator(
            model=self.cnn_model,
            device=self.device,
            criterion=nn.CrossEntropyLoss()
        )
        cnn_loss, cnn_metrics = cnn_evaluator.evaluate(self.test_loader)
        cnn_cm, cnn_roc = cnn_evaluator.detailed_analysis(self.test_loader, self.class_names)
        
        self.results_logger.log_evaluation_results('cnn', cnn_metrics)
        results['cnn'] = cnn_metrics
        
        # Evaluate ViT
        print("\nEvaluating ViT...")
        vit_evaluator = ModelEvaluator(
            model=self.vit_model,
            device=self.device,
            criterion=nn.CrossEntropyLoss()
        )
        vit_loss, vit_metrics = vit_evaluator.evaluate(self.test_loader)
        vit_cm, vit_roc = vit_evaluator.detailed_analysis(self.test_loader, self.class_names)
        
        self.results_logger.log_evaluation_results('vit', vit_metrics)
        results['vit'] = vit_metrics
        
        # Save confusion matrices and ROC curves
        if cnn_cm:
            cnn_cm.savefig(os.path.join(self.results_logger.save_dir, 'cnn_confusion_matrix.png'))
        if cnn_roc:
            cnn_roc.savefig(os.path.join(self.results_logger.save_dir, 'cnn_roc_curve.png'))
        if vit_cm:
            vit_cm.savefig(os.path.join(self.results_logger.save_dir, 'vit_confusion_matrix.png'))
        if vit_roc:
            vit_roc.savefig(os.path.join(self.results_logger.save_dir, 'vit_roc_curve.png'))
        
        return results
    
    def run_experiment(self):
        """Run complete experiment"""
        print("Starting Pneumonia Classification Experiment...")
        print(f"Device: {self.device}")
        
        # Setup data
        self.setup_data()
        
        # Create models
        self.create_models()
        
        # Train CNN
        cnn_history, cnn_time = self.train_cnn()
        
        # Train ViT
        vit_history, vit_time = self.train_vit()
        
        # Plot training curves for both models
        self.visualizer.plot_training_curves(
            cnn_history['train_loss'], cnn_history['val_loss'],
            cnn_history['train_acc'], cnn_history['val_acc']
        )
        
        # Evaluate both models
        test_results = self.evaluate_models()
        
        # Prepare results for comparison
        cnn_results = {
            'metrics': test_results['cnn'],
            'training_time': cnn_time,
            'num_params': self.cnn_model.get_num_params()
        }
        
        vit_results = {
            'metrics': test_results['vit'],
            'training_time': vit_time,
            'num_params': self.vit_model.get_num_params()
        }
        
        # Compare models
        comparison = self.comparator.compare_models(cnn_results, vit_results)
        
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"CNN Accuracy: {test_results['cnn']['accuracy']:.4f}")
        print(f"ViT Accuracy: {test_results['vit']['accuracy']:.4f}")
        print(f"Results saved to: {self.results_logger.save_dir}")
        
        return comparison


def main():
    """Main execution function"""
    config = {
        'dataset_path': './chest_xray',
        'batch_size': 32,
        'img_size': 224,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'val_split': 0.2,
        'early_stopping_patience': 10,
        'save_dir': './results',
        
        # ViT specific
        'vit_patch_size': 16,
        'vit_embed_dim': 768,
        'vit_depth': 12,
        'vit_num_heads': 12,
        
        # CNN specific
        'cnn_dropout': 0.5,
    }
    
    print("Starting Pneumonia Classification Experiment...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    experiment = ExperimentRunner(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()