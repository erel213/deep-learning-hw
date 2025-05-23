import pytest
import numpy as np
from utils import DataLoader

def test_dataloader_initialization():
    # Create sample data
    X = np.random.randn(10, 100)  # 10 features, 100 samples
    y = np.random.randn(100)      # 100 labels
    batch_size = 20
    
    # Initialize DataLoader
    dataloader = DataLoader(X, y, batch_size)
    
    # Test initialization
    assert dataloader.X.shape == (10, 100)
    assert dataloader.y.shape == (100,)
    assert dataloader.batch_size == 20
    assert len(dataloader.batches) == 5  # 100 samples / 20 batch_size = 5 batches


def test_dataloader_stop_iteration():
    # Create sample data
    X = np.random.randn(3, 6)  # 3 features, 6 samples
    y = np.random.randn(6)     # 6 labels
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
    y = np.array([1, 2, 3, 4, 5, 6])
    batch_size = 2
    
    # Initialize DataLoader
    dataloader = DataLoader(X, y, batch_size)
    
    # Test that batches contain correct data
    batches = list(dataloader)
    assert len(batches) == 3
    
    # Check that each batch contains the correct number of samples
    for X_batch, y_batch in batches:
        assert X_batch.shape[1] == y_batch.shape[0]
        assert X_batch.shape[0] == 2  # Number of features