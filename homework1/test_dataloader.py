import pytest
import numpy as np
from utils import DataLoader

def test_dataloader_initialization():
    # Create sample data
    X = np.random.randn(10, 100)  # 10 features, 100 samples
    y = np.random.randn(1, 100)   # 1 output, 100 samples
    batch_size = 20
    
    # Initialize DataLoader
    dataloader = DataLoader(X, y, batch_size)
    
    # Test initialization
    assert dataloader.X.shape == (10, 100)
    assert dataloader.y.shape == (1, 100)
    assert dataloader.batch_size == 20
    assert dataloader.n_batches == 5  # 100 samples / 20 batch_size = 5 batches
    assert len(dataloader) == 5

def test_dataloader_stop_iteration():
    # Create sample data
    X = np.random.randn(3, 6)  # 3 features, 6 samples
    y = np.random.randn(1, 6)  # 1 output, 6 samples
    batch_size = 2
    
    # Initialize DataLoader
    dataloader = DataLoader(X, y, batch_size)
    
    # Test that iteration stops after all batches
    batches = list(dataloader)
    assert len(batches) == 3  # 6 samples / 2 batch_size = 3 batches
    
    # Test that next iteration raises StopIteration
    with pytest.raises(StopIteration):
        next(dataloader)

def test_dataloader_batch_consistency():
    # Create sample data with known values
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    y = np.array([[1, 2, 3, 4, 5, 6]])  # Reshape to (1, 6)
    batch_size = 2
    
    # Initialize DataLoader
    dataloader = DataLoader(X, y, batch_size)
    
    # Test that batches contain correct data
    batches = list(dataloader)
    assert len(batches) == 3
    
    # Check that each batch contains the correct number of samples
    for X_batch, y_batch in batches:
        assert X_batch.shape[1] == batch_size  # Number of samples in batch
        assert X_batch.shape[0] == 2  # Number of features
        assert y_batch.shape[1] == batch_size  # Number of samples in batch
        assert y_batch.shape[0] == 1  # Number of outputs

def test_dataloader_reset():
    # Create sample data
    X = np.random.randn(3, 6)  # 3 features, 6 samples
    y = np.random.randn(1, 6)  # 1 output, 6 samples
    batch_size = 2
    
    # Initialize DataLoader
    dataloader = DataLoader(X, y, batch_size)
    
    # Get first set of batches
    first_batches = list(dataloader)
    assert len(first_batches) == 3
    
    # Reset and get second set of batches
    dataloader.reset()
    second_batches = list(dataloader)
    assert len(second_batches) == 3
    
    # Verify that we can iterate again after reset
    dataloader.reset()
    third_batches = list(dataloader)
    assert len(third_batches) == 3

def test_dataloader_iteration():
    # Create sample data
    X = np.random.randn(3, 6)  # 3 features, 6 samples
    y = np.random.randn(1, 6)  # 1 output, 6 samples
    batch_size = 2
    
    # Initialize DataLoader
    dataloader = DataLoader(X, y, batch_size)
    
    # Test multiple iterations
    for _ in range(3):
        batches = list(dataloader)
        assert len(batches) == 3
        for X_batch, y_batch in batches:
            assert X_batch.shape == (3, 2)  # 3 features, batch_size samples
            assert y_batch.shape == (1, 2)  # 1 output, batch_size samples