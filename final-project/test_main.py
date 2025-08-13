import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from main import (
    ImagePreprocessor, DatasetSplitter, PneumoniaDataLoader,
    PatchEmbedding, MultiHeadAttention, MLP, TransformerBlock,
    VisionTransformer, CNNModel, MetricsCalculator,
    TrainingVisualizer, ResultsLogger, ModelComparator,
    ModelTrainer, ModelEvaluator, ExperimentRunner
)


class TestImagePreprocessor:
    
    def setup_method(self):
        self.preprocessor = ImagePreprocessor(img_size=224, augment=True)
        self.preprocessor_no_aug = ImagePreprocessor(img_size=224, augment=False)
    
    def test_init(self):
        assert self.preprocessor.img_size == 224
        assert self.preprocessor.augment is True
        assert self.preprocessor_no_aug.augment is False
    
    def test_get_train_transforms(self):
        transform = self.preprocessor.get_train_transforms()
        assert transform is not None
        
        transform_no_aug = self.preprocessor_no_aug.get_train_transforms()
        assert transform_no_aug is not None
    
    def test_get_val_transforms(self):
        transform = self.preprocessor.get_val_transforms()
        assert transform is not None


class TestPatchEmbedding:
    
    def setup_method(self):
        self.patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
    
    def test_init(self):
        assert self.patch_embed.img_size == 224
        assert self.patch_embed.patch_size == 16
        assert self.patch_embed.embed_dim == 768
        assert self.patch_embed.num_patches == 196  # (224//16)^2
    
    def test_forward(self):
        x = torch.randn(2, 3, 224, 224)
        output = self.patch_embed(x)
        assert output.shape == (2, 196, 768)


class TestMultiHeadAttention:
    
    def setup_method(self):
        self.attention = MultiHeadAttention(embed_dim=768, num_heads=12)
    
    def test_init(self):
        assert self.attention.embed_dim == 768
        assert self.attention.num_heads == 12
        assert self.attention.head_dim == 64
    
    def test_forward(self):
        x = torch.randn(2, 197, 768)  # batch_size, seq_len, embed_dim
        output = self.attention(x)
        assert output.shape == (2, 197, 768)


class TestMLP:
    
    def setup_method(self):
        self.mlp = MLP(embed_dim=768, hidden_dim=3072)
    
    def test_forward(self):
        x = torch.randn(2, 197, 768)
        output = self.mlp(x)
        assert output.shape == (2, 197, 768)


class TestTransformerBlock:
    
    def setup_method(self):
        self.block = TransformerBlock(embed_dim=768, num_heads=12)
    
    def test_forward(self):
        x = torch.randn(2, 197, 768)
        output = self.block(x)
        assert output.shape == (2, 197, 768)


class TestVisionTransformer:
    
    def setup_method(self):
        self.vit = VisionTransformer(
            img_size=224, patch_size=16, num_classes=2,
            embed_dim=768, depth=12, num_heads=12
        )
    
    def test_init(self):
        assert self.vit.num_classes == 2
        assert self.vit.embed_dim == 768
    
    def test_forward(self):
        x = torch.randn(2, 3, 224, 224)
        output = self.vit(x)
        assert output.shape == (2, 2)
    
    def test_get_num_params(self):
        num_params = self.vit.get_num_params()
        assert isinstance(num_params, int)
        assert num_params > 0


class TestCNNModel:
    
    def setup_method(self):
        self.cnn = CNNModel(num_classes=2, dropout=0.5)
    
    def test_init(self):
        assert self.cnn.num_classes == 2
    
    def test_forward(self):
        x = torch.randn(2, 3, 224, 224)
        output = self.cnn(x)
        assert output.shape == (2, 2)
    
    def test_get_num_params(self):
        num_params = self.cnn.get_num_params()
        assert isinstance(num_params, int)
        assert num_params > 0


class TestMetricsCalculator:
    
    def setup_method(self):
        self.metrics = MetricsCalculator()
    
    def test_reset(self):
        self.metrics.reset()
        assert len(self.metrics.predictions) == 0
        assert len(self.metrics.targets) == 0
    
    def test_update(self):
        predictions = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([0, 1])
        
        self.metrics.update(predictions, targets)
        assert len(self.metrics.predictions) == 2
        assert len(self.metrics.targets) == 2
    
    def test_compute(self):
        # Create some dummy predictions and targets
        predictions = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
        targets = torch.tensor([0, 1, 0, 1])
        
        self.metrics.update(predictions, targets)
        result = self.metrics.compute()
        
        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1_score' in result
        
        # Check if values are in reasonable range
        for metric in result.values():
            assert 0.0 <= metric <= 1.0


class TestTrainingVisualizer:
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = TrainingVisualizer(save_dir=self.temp_dir)
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        assert self.visualizer.save_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
    
    def test_plot_training_curves(self):
        train_losses = [1.0, 0.8, 0.6, 0.4]
        val_losses = [1.1, 0.9, 0.7, 0.5]
        train_accs = [0.6, 0.7, 0.8, 0.9]
        val_accs = [0.5, 0.6, 0.7, 0.8]
        
        fig = self.visualizer.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        assert fig is not None
    
    def test_plot_model_comparison(self):
        cnn_metrics = {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1_score': 0.85}
        vit_metrics = {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.90, 'f1_score': 0.88}
        
        fig = self.visualizer.plot_model_comparison(cnn_metrics, vit_metrics)
        assert fig is not None


class TestResultsLogger:
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ResultsLogger(save_dir=self.temp_dir)
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        assert self.logger.save_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
    
    def test_log_evaluation_results(self):
        metrics = {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1_score': 0.85}
        self.logger.log_evaluation_results('test_model', metrics)
        
        # Check if files were created
        json_file = os.path.join(self.temp_dir, 'test_model_metrics.json')
        txt_file = os.path.join(self.temp_dir, 'test_model_summary.txt')
        
        assert os.path.exists(json_file)
        assert os.path.exists(txt_file)


class TestModelTrainer:
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cpu')
        self.model = CNNModel(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.trainer = ModelTrainer(
            model=self.model,
            device=self.device,
            criterion=self.criterion,
            optimizer=self.optimizer,
            save_dir=self.temp_dir
        )
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        assert self.trainer.device == self.device
        assert self.trainer.save_dir == self.temp_dir
    
    def test_save_checkpoint(self):
        metrics = {'accuracy': 0.85}
        self.trainer.save_checkpoint(epoch=5, metrics=metrics, is_best=True)
        
        checkpoint_path = os.path.join(self.temp_dir, 'checkpoint.pth')
        best_path = os.path.join(self.temp_dir, 'best_model.pth')
        
        assert os.path.exists(checkpoint_path)
        assert os.path.exists(best_path)


class TestModelEvaluator:
    
    def setup_method(self):
        self.device = torch.device('cpu')
        self.model = CNNModel(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()
        
        self.evaluator = ModelEvaluator(
            model=self.model,
            device=self.device,
            criterion=self.criterion
        )
    
    def test_init(self):
        assert self.evaluator.device == self.device
        assert self.evaluator.metrics is not None


class TestExperimentRunner:
    
    def setup_method(self):
        self.config = {
            'dataset_path': './test_data',
            'batch_size': 4,
            'img_size': 224,
            'num_epochs': 1,
            'learning_rate': 1e-4,
            'val_split': 0.2,
            'early_stopping_patience': 5,
            'save_dir': './test_results',
            'vit_patch_size': 16,
            'vit_embed_dim': 768,
            'vit_depth': 1,  # Reduced for testing
            'vit_num_heads': 12,
            'cnn_dropout': 0.5,
        }
        
        self.runner = ExperimentRunner(self.config)
    
    def test_init(self):
        assert self.runner.config == self.config
        assert self.runner.device is not None
        assert self.runner.results_logger is not None
        assert self.runner.visualizer is not None
        assert self.runner.comparator is not None
    
    def test_create_models(self):
        self.runner.create_models()
        
        assert self.runner.cnn_model is not None
        assert self.runner.vit_model is not None
        assert isinstance(self.runner.cnn_model, CNNModel)
        assert isinstance(self.runner.vit_model, VisionTransformer)


class TestIntegration:
    """Integration tests for model forward passes and basic functionality"""
    
    def test_cnn_forward_pass(self):
        model = CNNModel(num_classes=2)
        x = torch.randn(1, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()
    
    def test_vit_forward_pass(self):
        model = VisionTransformer(
            img_size=224, patch_size=16, num_classes=2,
            embed_dim=256, depth=2, num_heads=8  # Smaller for testing
        )
        x = torch.randn(1, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()
    
    def test_patch_embedding_shape_consistency(self):
        """Test that patch embedding produces consistent shapes"""
        patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
        
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = patch_embed(x)
            expected_shape = (batch_size, 196, 768)
            assert output.shape == expected_shape
    
    def test_attention_output_shapes(self):
        """Test multi-head attention maintains correct shapes"""
        attention = MultiHeadAttention(embed_dim=768, num_heads=12)
        
        for seq_len in [50, 100, 197]:
            x = torch.randn(2, seq_len, 768)
            output = attention(x)
            assert output.shape == (2, seq_len, 768)


# Pytest fixtures for common test data
@pytest.fixture
def sample_tensor():
    """Sample tensor for testing"""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def temp_directory():
    """Create and cleanup temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing"""
    return {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1_score': 0.85}


# Additional parametrized tests
@pytest.mark.parametrize("img_size,patch_size,expected_patches", [
    (224, 16, 196),
    (224, 32, 49),
    (256, 16, 256),
])
def test_patch_embedding_sizes(img_size, patch_size, expected_patches):
    """Test patch embedding with different image and patch sizes"""
    patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=768)
    assert patch_embed.num_patches == expected_patches


@pytest.mark.parametrize("embed_dim,num_heads", [
    (768, 12),
    (512, 8),
    (256, 4),
])
def test_attention_different_configs(embed_dim, num_heads):
    """Test attention with different embedding dimensions and head counts"""
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(2, 50, embed_dim)
    output = attention(x)
    assert output.shape == (2, 50, embed_dim)


@pytest.mark.parametrize("num_classes", [2, 5, 10])
def test_cnn_different_classes(num_classes):
    """Test CNN with different number of output classes"""
    model = CNNModel(num_classes=num_classes)
    x = torch.randn(1, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (1, num_classes)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_vit_different_batch_sizes(batch_size):
    """Test ViT with different batch sizes"""
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=2,
        embed_dim=256, depth=2, num_heads=8
    )
    x = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 2)


if __name__ == '__main__':
    # Run pytest programmatically
    pytest.main([__file__, '-v', '--tb=short'])