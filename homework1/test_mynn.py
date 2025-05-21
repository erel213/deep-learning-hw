import pytest
import numpy as np
from utils import MyNN

@pytest.fixture
def simple_nn():
    """Fixture that creates a simple neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron."""
    return MyNN(learning_rate=0.1, layer_sizes=[2, 3, 1])

@pytest.fixture
def sample_batch():
    """Fixture that creates a sample batch of data."""
    # Create a batch of 4 instances, each with 2 features
    X = np.array([[0.1, 0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7, 0.8]])
    # Create corresponding labels
    y = np.array([[0, 1, 0, 1]])
    return X, y

def test_update(simple_nn):
    """Test the update function to ensure weights and biases are updated correctly."""
    # Set some initial gradients
    simple_nn.grads = {
        'dW_1': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        'db_1': np.array([0.1, 0.2, 0.3]),
        'dW_2': np.array([[0.1, 0.2, 0.3]]),
        'db_2': np.array([0.1])
    }
    
    # Store original weights and biases
    original_W1 = simple_nn.model_params['W_1'].copy()
    original_b1 = simple_nn.model_params['b_1'].copy()
    original_W2 = simple_nn.model_params['W_2'].copy()
    original_b2 = simple_nn.model_params['b_2'].copy()
    
    # Perform update
    simple_nn.update()
    
    # Check if weights and biases were updated correctly
    assert np.allclose(simple_nn.model_params['W_1'], 
                      original_W1 - 0.1 * simple_nn.grads['dW_1'])
    assert np.allclose(simple_nn.model_params['b_1'], 
                      original_b1 - 0.1 * simple_nn.grads['db_1'])
    assert np.allclose(simple_nn.model_params['W_2'], 
                      original_W2 - 0.1 * simple_nn.grads['dW_2'])
    assert np.allclose(simple_nn.model_params['b_2'], 
                      original_b2 - 0.1 * simple_nn.grads['db_2'])

def test_forward_batch(simple_nn, sample_batch):
    """Test the forward_batch function to ensure it processes batches correctly."""
    X, _ = sample_batch
    
    # Get batch output
    batch_output = simple_nn.forward_batch(X)
    
    # Check output shape
    assert batch_output.shape == (1, 4)  # 1 output neuron, 4 instances
    
    # Check that all outputs are between 0 and 1 (due to sigmoid activation)
    assert np.all(batch_output >= 0) and np.all(batch_output <= 1)
    
    # Check that outputs are different for different inputs
    assert not np.allclose(batch_output[:, 0], batch_output[:, 1])

def test_backward_batch(simple_nn, sample_batch):
    """Test the backward_batch function to ensure it computes gradients correctly."""
    X, y = sample_batch
    
    # Perform forward pass first
    simple_nn.forward_batch(X)
    
    # Perform backward pass
    simple_nn.backward_batch(y)
    
    # Check that gradients were computed for all layers
    assert 'dW_1' in simple_nn.grads
    assert 'db_1' in simple_nn.grads
    assert 'dW_2' in simple_nn.grads
    assert 'db_2' in simple_nn.grads
    
    print(simple_nn.grads)
    # Check gradient shapes
    assert simple_nn.grads['dW_1'].shape == (3, 2)  # 3 hidden neurons, 2 input features
    assert simple_nn.grads['db_1'].shape == (2,)    # 3 hidden neurons
    assert simple_nn.grads['dW_2'].shape == (1, 3)  # 1 output neuron, 3 hidden neurons
    assert simple_nn.grads['db_2'].shape == (3,)    # 1 output neuron

def test_log_loss_batch(simple_nn, sample_batch):
    """Test the log_loss_batch function to ensure it computes batch loss correctly."""
    X, y = sample_batch
    
    # Get predictions
    y_hat = simple_nn.forward_batch(X)
    
    # Compute batch loss
    batch_loss = simple_nn.log_loss_batch(y_hat, y)
    
    # Check that loss is a vector with size one that contain scalar
    assert batch_loss.shape == (1,)
    assert isinstance(batch_loss[0], (int, float))
    
    # Check that loss is non-negative assert batch_loss >= 0
    assert batch_loss[0] >= 0

    # Check that loss is finite
    assert np.isfinite(batch_loss[0])

