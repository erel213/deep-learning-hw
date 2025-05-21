import numpy as np
np.random.seed(42)

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
      dz = (a_l_1 * (1 - a_l_1)).reshape(-1, 1) * np.dot(W_l.T, dz.reshape(-1, 1))
      # TODO: calculate and memorize db as well.
      db = dz.flatten() # Gradient of the biases for the current layer - flatten() returns a copy of an array collapsed into 1D (a flat vector).
      self.grads['db_' + str(layer_index)] = db

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


nn = MyNN(0.01, [3, 2, 1])

def train(X, y, epochs, batch_size):
  '''
  Train procedure, please note the TODOs inside
  '''
  for e in range(1, epochs + 1):
    epoch_loss = 0
    # TODO: shuffle
    batches = np.random.permutation(X.shape[1])
    #... TODO: divide to batches
    for X_b, y_b in batches:
      y_hat = nn.forward_batch(X_b)
      epoch_loss += nn.log_loss_batch(y_hat, y_b)
      nn.backward_batch(y_b)
      nn.update()
    print(f'Epoch {e}, loss={epoch_loss/len(batches)}')



# TODO: Preprocess the bike sharing dataset ('hour.csv')
# - Load the dataset from the provided hour.csv file
# - Select the required features (temp, atemp, hum, windspeed, weekday)
# - Extract the target variable (success)
# - Normalize/standardize features if necessary
# - Split the data into training (75%), validation (10%), and test (15%) sets
# - Create DataLoader objects with batch_size=8



# TODO: Train the neural network
# - Implement the network with architecture [5, 40, 30, 10, 7, 5, 3, 1]
# - Train for exactly 100 epochs on the training set
# - Use batch_size=8 as specified
# - Calculate and store train and validation loss for each epoch
# - Track training progres



# TODO: Create visualizations of the learning process
# - Plot the training loss per epoch
# - Create additional relevant plots (validation loss, learning curves, etc.)
# - Make sure all plots have proper labels, titles, and legends
# - Add brief analysis of what the plots reveal about your model's performance


# TODO: Evaluate model performance on the test set
# - Calculate and report the loss on the test set
# - Calculate and report the accuracy on the test set
# - Compare test performance with training/validation performance
# - Analyze model strengths and weaknesses
# - Discuss any overfitting/underfitting issues observed