import numpy as np
import pandas as pd
np.random.seed(42)
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class MyNN:
  def __init__(self, learning_rate, layer_sizes):
    '''
    learning_rate - the learning to use in backward
    layer_sizes - a list of numbers, each number repreents the nuber of neurons
                  to have in every layer. Therfore, the length of the list
                  represents the number layers this network has.
    '''
    self.learning_rate = learning_rate

    # layer_sizes: A list like [3, 5, 1] meaning:
      # Input layer with 3 neurons
      # hidden layer with 5 neurons
      # Output layer with 1 neuron

    self.layer_sizes = layer_sizes


    self.model_params = {}
    self.memory = {}
    self.grads = {}

    # Initializing weights
    for layer_index in range(len(layer_sizes) - 1):
      W_input = layer_sizes[layer_index + 1]
      W_output = layer_sizes[layer_index]
      self.model_params['W_' + str(layer_index + 1)] = np.random.randn(W_input, W_output) * 0.1
      self.model_params['b_' + str(layer_index + 1)] = np.random.randn(W_input) * 0.1

  # performs forward propagation through the neural network for a single input instance.
  # This function takes a single input x (a 1D NumPy array) and returns the final output (i.e., prediction).
  def forward_single_instance(self, x):
    a_i_1 = x                                                     # a_i_1 is the activation from the previous layer (starting with input x).
    self.memory['a_0'] = x                                        # The input is saved in self.memory for use during backpropagation
    for layer_index in range(len(self.layer_sizes) - 1):          # Loop over layers - Iterates through all weight layers (i.e., from input → hidden → output).

      # If layer_sizes = [3, 5, 1], the loop runs twice:
      #   Layer 1: from input (3) to hidden (5)
      #   Layer 2: from hidden (5) to output (1)

      # Retrieves the weight matrix W_i and bias vector b_i for the current layer:
      W_i = self.model_params['W_' + str(layer_index + 1)]
      b_i = self.model_params['b_' + str(layer_index + 1)]

      # Linear transformation - Zi = W_i * a_i_1 + b_i
      # a_i_1 is the activation from the previous layer (starting with input x).
      # W_i is the weight matrix for the current layer.
      # b_i is the bias vector for the current layer.
      # The dot product of W_i and a_i_1 is computed, and the bias b_i is added to it.
      z_i = np.dot(W_i, a_i_1) + b_i
      # Activation function - Ai = sigmoid(Zi)
      # The sigmoid activation function is applied to the linear transformation result (z_i).
      # The sigmoid function is defined as: sigmoid(z) = 1 / (1 + exp(-z))
      # This function squashes the output to a range between 0 and 1.
      # The result is stored in a_i, which represents the activation of the current layer.
      a_i = 1/(1+np.exp(-z_i))

      # The activation a_i is saved in self.memory for use during backpropagation.
      self.memory['a_' + str(layer_index + 1)] = a_i

      # The activation a_i becomes the input for the next layer (i.e., a_i_1 for the next iteration).
      # "a_i_1" means a_{i-1}, i.e., activation from the previous layer
      a_i_1 = a_i
    return a_i_1

  # binary cross-entropy loss (also called logistic loss) for a single prediction.
  # y=1 y_hat=0 => true
  # y=0 y_hat=1 => true
  # y=y_hat     => false
  def log_loss(self, y_hat, y):
    '''
    Logistic loss, assuming a single value in y_hat and y.
    '''
    m = y_hat[0]
    cost = -y[0]*np.log(y_hat[0]) - (1 - y[0])*np.log(1 - y_hat[0])
    return cost

  # Performs backpropagation through the neural network for a single input instance.
  # This function takes a single target value y (a 1D NumPy array) and computes the gradients.
  # It computes gradients of the weights (and should also compute gradients of biases) to be used later in weight updates.
  def backward_single_instance(self, y):
    a_output = self.memory['a_' + str(len(self.layer_sizes) - 1)]           # The output of the last layer (i.e., the prediction). final output of the network = y_hat
    dz = a_output - y               # The difference between the predicted output (a_output) and the true target value (y). derivative of loss w.r.t. 

    for layer_index in range(len(self.layer_sizes) - 1, 0, -1): #  Loop through layers in reverse order, if you have 3 layers ([3, 4, 1]), layer_index will be: 2 → 1
      print(layer_index)
      a_l_1 = self.memory['a_' + str(layer_index - 1)]  # the activation from the previous layer (i.e., a_{i-1}).
      dW = np.dot(dz.reshape(-1, 1), a_l_1.reshape(1, -1))  # Gradient of the weights for the current layer.
      self.grads['dW_' + str(layer_index)] = dW
      W_l = self.model_params['W_' + str(layer_index)]
      # TODO: calculate and memorize db as well.
      db = dz.reshape(-1)
      self.grads['db_' + str(layer_index)] = db
      dz = (a_l_1 * (1 - a_l_1)).reshape(-1, 1) * np.dot(W_l.T, dz.reshape(-1, 1)) # WARNING: I change the order here initially we first compute dz and the update db 



  # TODO: update weights with grads
  def update(self):
    """
    Updates weights and biases using the computed gradients and learning rate.
    Uses gradient descent to update parameters.
    """
    for layer_index in range(1, len(self.layer_sizes)):
        # Get weight and bias gradients
        dW = self.grads.get('dW_' + str(layer_index), 0)
        db = self.grads.get('db_' + str(layer_index), 0)
        
        # Get current weights and biases
        W = self.model_params['W_' + str(layer_index)]
        b = self.model_params['b_' + str(layer_index)]
        
        # Update weights and biases using gradient descent
        self.model_params['W_' + str(layer_index)] = W - self.learning_rate * dW
        self.model_params['b_' + str(layer_index)] = b - self.learning_rate * db

  # TODO: implement forward for a batch X.shape = (network_input_size, number_of_instance)
  def forward_batch(self, X):
    """
    Performs forward propagation for a batch of input instances.
    
    Parameters:
    X -- Input data, shape: (network_input_size, number_of_instances)
    
    Returns:
    batch_output -- Output predictions, shape: (output_size, number_of_instances)
    """
    # Number of instances in the batch
    number_of_instances = X.shape[1]
    # Output size of the network (number of neurons in the last layer)
    output_size = self.layer_sizes[-1]
    
    # Initialize output array
    batch_output = np.zeros((output_size, number_of_instances))
    
    # Process each instance in the batch
    for i in range(number_of_instances):
        # Extract the i-th instance
        x_i = X[:, i]
        # Forward pass for the i-th instance
        instance_output = self.forward_single_instance(x_i)
        # Store the output of the i-th instance
        batch_output[:, i] = instance_output
    
    return batch_output

  # TODO: implement backward for a batch y.shape = (1, number_of_instance)
  def backward_batch(self, y):
    """
    Performs backpropagation for a batch of target values.
    Computes gradients that are averaged across all instances.
    
    Parameters:
    y -- Target values, shape: (1, number_of_instances)
    """
    # Number of instances in the batch
    number_of_instances = y.shape[1]
    
    # Initialize gradients storage for accumulation
    grads_sum = {}
    
    # Process each instance in the batch
    for i in range(number_of_instances):
        # Extract the i-th target
        y_i = y[:, i]
        
        # Backward pass for the i-th instance
        self.backward_single_instance(y_i)
        
        # Accumulate gradients
        for key in self.grads:
            if key not in grads_sum:
                grads_sum[key] = self.grads[key]
            else:
                grads_sum[key] += self.grads[key]
    
    # Average the accumulated gradients
    for key in grads_sum:
        self.grads[key] = grads_sum[key] / number_of_instances

  # TODO: implement log_loss_batch, for a batch of instances
  def log_loss_batch(self, y_hat, y):
    """
    Computes the average binary cross-entropy loss for a batch of predictions.
    
    Parameters:
    y_hat -- Predicted outputs, shape: (output_size, number_of_instances)
    y -- Target values, shape: (output_size, number_of_instances)
    
    Returns:
    cost -- Average loss across all instances
    """
    # Number of instances in the batch
    number_of_instances = y.shape[1]
    
    # Initialize total cost
    total_cost = 0
    
    # Compute loss for each instance
    for i in range(number_of_instances):
        # Extract the i-th prediction and target
        y_hat_i = y_hat[:, i].reshape(-1, 1)
        y_i = y[:, i].reshape(-1, 1)
        
        # Compute loss for the i-th instance
        instance_cost = self.log_loss(y_hat_i, y_i)
        
        # Accumulate total cost
        total_cost += instance_cost
    
    # Compute average cost
    cost = total_cost / number_of_instances
    
    return cost



def train(X, y, epochs, batch_size):
  '''
  Train procedure, please note the TODOs inside
  '''
  for e in range(1, epochs + 1):
    epoch_loss = 0
    # TODO: shuffle
    batches = np.random.permutation(X.shape[1])
    #... TODO: divide to batches
    batch_indices = np.array_split(batches, X.shape[1] // batch_size)
    
    for batch_idx in batch_indices:
        # Use indices to select the corresponding data
        X_b = X[:, batch_idx]
        y_b = y[:, batch_idx]
        
        y_hat = nn.forward_batch(X_b)
        epoch_loss += nn.log_loss_batch(y_hat, y_b)
        nn.backward_batch(y_b)
        nn.update()
    print(f'Epoch {e}, loss={epoch_loss/len(batch_indices)}')



# TODO: Preprocess the bike sharing dataset ('hour.csv')
# - Load the dataset from the provided hour.csv file
# - Select the required features (temp, atemp, hum, windspeed, weekday)
# - Extract the target variable (success)
# - Normalize/standardize features if necessary
# - Split the data into training (75%), validation (10%), and test (15%) sets
# - Create DataLoader objects with batch_size=8
class DataLoader:
  def __init__(self, X, y, batch_size):
    self.X = X
    self.y = y
    self.batch_size = batch_size
    self.indices = np.random.permutation(X.shape[1])
    self.batches = np.array_split(self.indices, self.batch_size)

  def __iter__(self):
    return self
  
  def __next__(self):
    if len(self.batches) == 0:
      raise StopIteration
    batch_indices = self.batches.pop(0)
    X_batch = self.X[:, batch_indices]
    y_batch = self.y[batch_indices]
    return X_batch, y_batch

def preprocess_data(file_path):
    '''
    Preprocess the bike sharing dataset ('hour.csv')
    '''
    # Load numpy arrays
    df = pd.read_csv(file_path)
    
    # Define the features we want to use
    feature_columns = ['temp', 'atemp', 'hum', 'windspeed', 'weekday']
    
    # Create a mask for the features
    feature_mask = df.columns.isin(feature_columns)
    
    # Select features using the mask
    X = df.loc[:, feature_mask].values
    y = df['success'].values
    
    # Normalize/standardize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Split data into training, validation, and test sets
    train_X, train_y, val_X, val_y, test_X, test_y = split_data(X, y)
    
    # Create DataLoader objects
    train_loader = DataLoader(train_X, train_y, batch_size=8)
    val_loader = DataLoader(val_X, val_y, batch_size=8)
    test_loader = DataLoader(test_X, test_y, batch_size=8)
    
    return train_loader, val_loader, test_loader


def split_data(X, y, train_ratio=0.75, val_ratio=0.1, test_ratio=0.15):
  '''
  Split the data into training, validation, and test sets
  X - numpy array of shape (num_instances, num_features)
  y - numpy array of shape (num_instances,)
  train_ratio - float, the ratio of the training set
  val_ratio - float, the ratio of the validation set
  test_ratio - float, the ratio of the test set
  Returns:
  train_X, train_y, val_X, val_y, test_X, test_y
  '''
  # Calculate the number of instances for each set
  num_instances = X.shape[0]
  num_train = int(train_ratio * num_instances)
  num_val = int(val_ratio * num_instances)

  # Shuffle the data
  indices = np.random.permutation(num_instances)
  X = X[indices]
  y = y[indices]

  # Split the data
  train_X = X[:num_train]
  train_y = y[:num_train]
  val_X = X[num_train:num_train+num_val]
  val_y = y[num_train:num_train+num_val]
  test_X = X[num_train+num_val:]
  test_y = y[num_train+num_val:]

  return train_X, train_y, val_X, val_y, test_X, test_y

# TODO: Train the neural network
# - Implement the network with architecture [5, 40, 30, 10, 7, 5, 3, 1]
# - Train for exactly 100 epochs on the training set
# - Use batch_size=8 as specified
# - Calculate and store train and validation loss for each epoch
# - Track training progres
def train_nn(train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    """
    Train the neural network with visualization support.
    
    Parameters:
    train_loader: DataLoader object for the training set
    val_loader: DataLoader object for the validation set
    epochs: int, the number of epochs to train the network
    """
    nn = MyNN(0.01, [5, 40, 30, 10, 7, 5, 3, 1])
    visualizer = TrainingVisualizer()
    
    for epoch in range(epochs):
        # Training phase
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_hat = nn.forward_batch(X_batch)
            train_loss += nn.log_loss_batch(y_hat, y_batch)
            
            # Calculate accuracy
            predictions = (y_hat > 0.5).astype(int)
            train_correct += np.sum(predictions == y_batch)
            train_total += y_batch.size
            
            # Backward pass and update
            nn.backward_batch(y_batch)
            nn.update()
        
        # Calculate average training loss and accuracy
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        for X_batch, y_batch in val_loader:
            y_hat = nn.forward_batch(X_batch)
            val_loss += nn.log_loss_batch(y_hat, y_batch)
            
            # Calculate accuracy
            predictions = (y_hat > 0.5).astype(int)
            val_correct += np.sum(predictions == y_batch)
            val_total += y_batch.size
        
        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update visualizer
        visualizer.update(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=nn.learning_rate
        )
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Generate visualizations
    visualizer.plot_training_curves()
    visualizer.plot_learning_rate()
    visualizer.plot_loss_distribution()
    
    # Print training summary
    summary = visualizer.generate_summary()
    print("\nTraining Summary:")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    
    return nn, visualizer



# TODO: Create visualizations of the learning process
# - Plot the training loss per epoch
# - Create additional relevant plots (validation loss, learning curves, etc.)
# - Make sure all plots have proper labels, titles, and legends
# - Add brief analysis of what the plots reveal about your model's performances

class TrainingVisualizer:
    def __init__(self):
        """Initialize the training visualizer with empty history."""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def update(self, 
              train_loss: float, 
              val_loss: float, 
              train_accuracy: float = None, 
              val_accuracy: float = None,
              learning_rate: float = None):
        """Update the training history with new metrics."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        if train_accuracy is not None:
            self.history['train_accuracy'].append(train_accuracy)
        if val_accuracy is not None:
            self.history['val_accuracy'].append(val_accuracy)
        if learning_rate is not None:
            self.history['learning_rates'].append(learning_rate)
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(12, 5))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss', color='blue')
        plt.plot(self.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy curves if available
        if self.history['train_accuracy'] and self.history['val_accuracy']:
            plt.subplot(1, 2, 2)
            plt.plot(self.history['train_accuracy'], label='Training Accuracy', color='blue')
            plt.plot(self.history['val_accuracy'], label='Validation Accuracy', color='red')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_learning_rate(self, save_path: str = None):
        """Plot learning rate changes over time."""
        if not self.history['learning_rates']:
            return
        
        plt.figure(figsize=(8, 4))
        plt.plot(self.history['learning_rates'], color='green')
        plt.title('Learning Rate Changes')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_loss_distribution(self, save_path: str = None):
        """Plot the distribution of training and validation losses."""
        plt.figure(figsize=(10, 4))
        
        # Plot histograms
        plt.subplot(1, 2, 1)
        plt.hist(self.history['train_loss'], bins=30, alpha=0.5, label='Training Loss', color='blue')
        plt.hist(self.history['val_loss'], bins=30, alpha=0.5, label='Validation Loss', color='red')
        plt.title('Loss Distribution')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot box plots
        plt.subplot(1, 2, 2)
        plt.boxplot([self.history['train_loss'], self.history['val_loss']], 
                   labels=['Training Loss', 'Validation Loss'])
        plt.title('Loss Box Plot')
        plt.ylabel('Loss Value')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def generate_summary(self) -> Dict:
        """Generate a summary of the training metrics."""
        summary = {
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'min_train_loss': min(self.history['train_loss']),
            'min_val_loss': min(self.history['val_loss']),
            'epochs_trained': len(self.history['train_loss'])
        }
        
        if self.history['train_accuracy'] and self.history['val_accuracy']:
            summary.update({
                'final_train_accuracy': self.history['train_accuracy'][-1],
                'final_val_accuracy': self.history['val_accuracy'][-1],
                'max_train_accuracy': max(self.history['train_accuracy']),
                'max_val_accuracy': max(self.history['val_accuracy'])
            })
        
        return summary


# TODO: Evaluate model performance on the test set
# - Calculate and report the loss on the test set
# - Calculate and report the accuracy on the test set
# - Compare test performance with training/validation performance
# - Analyze model strengths and weaknesses
# - Discuss any overfitting/underfitting issues observed
