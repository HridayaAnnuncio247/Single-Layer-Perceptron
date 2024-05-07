import numpy as np
data = np.array([[1,2,3], [2,4,6], [3,6,9]])# row wise data, 3rd column is output value

def normalize(inputs):
  pass
def initialize_weights(row):
  """
  row: one row of predictor values of the data.
  :return: list (n,1)
  """
  weights = np.random.rand(len(row), 1) # random wts between 0 and 1. Random values uniformly chosen
  # using transpose of weights to help with matrix multiplication in later steps
  return weights


def activate(input_sum, activation_function = "Logistic Regression"):
  """
  ndarray (m,1) input_sum: the dot product of inputs and their corresponding weights
  str activation_function: name of activation function to be used on the input_sum.
  :return: Float value
  """
  if activation_function == "Logistic Regression":
    activated_value = 1/(1 + np.exp(-input_sum)) # using broadcasting
  return activated_value


def forward_pass(inputs, weights, activation_function):
  """
  np.array (m, n) inputs: inputs rows of predictor values of the data
  np.array (n, 1) weights:
  str activation_function: name of activation function to be used on the input_sum.
  :return: Float  value
  """
  #inputs = np.multiply(weights, row)
  # need dot product as need to multiply element wise and then add products to subsequently pass through activation function
  #input_sum = np.sum(np.multiply(weights, row)) # changing this to matrix multiplication so that batch input processing can be done
  input_sum = np.matmul(inputs, weights)#result is (m,1) nd array
  activated_output = activate(activation_function, input_sum)
  return activated_output#also the predicted output

def calculate_error(predicted_output, actual_output, type_of_error):
  """
  np.array(m,1) predicted_output: predicted outputs for each data row.
  np.array(m,1) actual_output: actual output for each row of data
  str type_of_error: the kind of error that has to be caclulated -
                      (1) SSE -  Sum of Squared Error
  :return: float
  """
  if type_of_error == "SSE":
    error = np.subtract(predicted_output, actual_output) # difference between predicted and actual outputs
    squared_error = np.square(error) #squaring each error
    sum_of_squares =np.sum(squared_error)
    return sum_of_squares



def backward_pass(input, predicted_output, actual_output, learning_rate = 0.1, activation_function = "SSE"):
  """
  np.array(m,n) input
  np.array(m,1) predicted_output: predicted outputs for each data row.
  np.array(m,1) actual_output: actual output for each row of data
  float learning_rate: learning rate i.e. rate at which changes are made
  str error_type: name of the type of error used.
  :return: numpy array of shape (n,1)
  """
  if activation_function == "SSE":
    term1 =np.subtract(np.ones((1,len(predicted_output))), predicted_output)
    term2 = np.subtract(actual_output, predicted_output)
    delta = np.multiply(predicted_output, np.multiply(term1, term2))
    learning_rate_array = np.full(len(predicted_output), learning_rate)
    delw = np.multiply(learning_rate_array, np.multiply(delta, input))
    delw = np.sum(delw, axis = 0)# needs editing as del w is currently m X n. We need sums of separate weight (m weights) to average for each of the n weights.Therefore adding every column
    delw = delw / len(predicted_output) # evaluating the mean of the change required in each weight w.r.t all the inputs. Broadcasting is being used over here
    return delw.transpose()
def change_weights(delw, weights):
  """
  ap.array(n,1) delw: the change to be made in the weights.
  np.array(n,1) weights: The current weights whose values have to be changed.
  :return: np.array(m,1) of the new weights.
  """
  #delw_array = np.full(len(weights[0]), delw)
  new_weights = np.subtract(weights, delw)
  return new_weights

def epochs(input, actual_output, threshold=0.1):
  """
  np.array(m,n) input
  np.array(m,1) actual_output: actual output for each row of data
  float threshold : The difference between 2 epochs that should make the iterations stop after a minimum value has been reached.
  :return: numpy array (n,1) of the best weights
  """
  ##need to add - default values for threshold, activation func, etc.
  weights = initialize_weights(input[0])
  epch = 1
  difference = 0
  previous_error = 0
  while -threshold > difference or difference > threshold:
    print("Epoch number:", epch)
    predicted_output = forward_pass(input, weights)
    error = calculate_error(predicted_output, actual_output)
    print("error:", error)
    delw = backward_pass(input, predicted_output, actual_output, weights, error )
    weights = change_weights(delw, weights) # changing weights to the new values of weights.
    difference = error - previous_error
    previous_error = error
  return weights # returning the most efficient weights to use later for prediction.


if __name__ == "__main__":

  pass