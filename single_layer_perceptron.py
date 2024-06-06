import numpy as np
import pandas as pd
import time

data = np.array([[1,2,3], [2,4,6], [3,6,9]])# row wise data, 3rd column is output value


def normalize(inputs, outputs, type_of_normalization):
  """
  ndarray (m,n) inputs: The predictor values.
  ndarray (m,1) outputs: Predictand value.
  str type_of_normalization: Describes the kind of normalization to be applied
                             to the inputs and outputs.
                             Possible types: min_max, z_score (to be added soon).
  :return: normalized input and output values.
  """
  x = inputs.transpose()
  y = outputs.transpose()

  if type_of_normalization == "min_max":

    min_values = np.array([[np.min(i)] for i in x] )
    max_values = np.array([[np.max(i)] for i in x] )
    normalized_inputs = (x - min_values) / (max_values - min_values)
    #normalized_inputs = normalized_inputs.transpose()
    term1 = min_output = np.min(y[0])
    term2 = max_output = np.max(y[0])
    normalized_outputs = (y - min_output)/(max_output - min_output)
    #normalized_outputs = normalized_outputs.transpose()
    # need to also return min and mac values for de normalization

  elif type_of_normalization == "z_score":
    mean = np.reshape(np.mean(inputs, axis = 0), (len(inputs[0]),1)) # finding column wise means
    stds = np.reshape(np.std(inputs, axis = 0), (len(inputs[0]),1))
    term1 = mean_output = np.mean(outputs, axis = 0) # finding column wise means
    term2 = stds_output = np.std(outputs, axis = 0)
    normalized_inputs = (x - mean)/stds
    normalized_outputs = (y-mean_output)/stds_output

  return normalized_inputs.transpose(), normalized_outputs.transpose(),[term1, term2]

def denormalize(outputs, term1, term2, type_of_deormalization):
  """
  ndarray (m,1) outputs: Predicted outputs that have to be brought back to
                         the main form so as to prit out the prediction.
  float term1: Either the min value or the mean.
  float term2: Either the max value or the standard deviation.
  str type_of_denormalization: Type of denormalization that has to be done -min_max or z_score
  :return: ndarray of denormalized values.
  """
  if type_of_denormalization == "min_max":
    result = outputs * (term2 - term1) + term1
  elif type_of_denormalization == "z_score":
    result = output * term2 + term1
  return result

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
    #print("exponential input sum:", np.exp(-input_sum) )
    activated_value = 1/(1 + np.exp(-input_sum)) # using broadcasting
  return activated_value


def forward_pass(inputs, weights, activation_function="Logistic Regression"):
  """
  np.array (m, n) inputs: inputs rows of predictor values of the data
  np.array (n, 1) weights:
  str activation_function: name of activation function to be used on the input_sum.
  :return: Float  value
  """
  #inputs = np.multiply(weights, row)
  # need dot product as need to multiply element wise and then add products to subsequently pass through activation function
  #input_sum = np.sum(np.multiply(weights, row)) # changing this to matrix multiplication so that batch input processing can be done
  #print("weights shape", weights.shape)
  input_sum = np.matmul(inputs, weights)#result is (m,1) nd array
  #print("input sum mX1:", input_sum)
  activated_output = activate(input_sum, activation_function)
  #print("activated output:", activated_output)
  return activated_output#also the predicted output

def calculate_error(predicted_output, actual_output, type_of_error = "SSE"):
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



def backward_pass(input, predicted_output, actual_output, learning_rate = 0.1, error_type = "SSE"):
  """
  np.array(m,n) input
  np.array(m,1) predicted_output: predicted outputs for each data row.
  np.array(m,1) actual_output: actual output for each row of data
  float learning_rate: learning rate i.e. rate at which changes are made
  str error_type: name of the type of error used.
  :return: numpy array of shape (n,1)
  """
  if error_type == "SSE":

    term1 =np.subtract(np.ones((len(predicted_output),1)), predicted_output)
    #print("predicted outputs:", predicted_output) # problem found that predicted outputs were all 1
    #print("ones:",np.ones((len(predicted_output),1)) )
    term2 = np.subtract(actual_output, predicted_output)
    #print("ones shape:", np.ones((1,len(predicted_output))).shape)
    #print("term 1 shape:", term1.shape)
    #print("term 2 shape:", term2.shape)
    #print("term1:", term1)
    #print("term2:", term2)

    delta = np.multiply(predicted_output, np.multiply(term1, term2))
    #print("delta shape", delta.shape)
    #print("predicted output shape:", predicted_output.shape)
    #print("input shape:", input.shape)
    #print("delta:", delta)
    learning_rate_array = np.full(len(predicted_output), learning_rate).reshape(len(predicted_output),1) #Reshaped to 2D array
    #print("learning rate array shape", learning_rate_array.shape)
    delw = np.multiply(learning_rate_array, np.multiply(delta, input))
    delw = np.sum(delw, axis = 0)# needs editing as del w is currently m X n. We need sums of separate weight (m weights) to average for each of the n weights.Therefore adding every column
    delw = delw / len(predicted_output) # evaluating the mean of the change required in each weight w.r.t all the inputs. Broadcasting is being used over here
    #print("delw before returning:", delw)
    #print("transpose", delw, delw.transpose)
    return delw.reshape((len(delw),1))
def change_weights(delw, weights):
  """
  ap.array(n,1) delw: the change to be made in the weights.
  np.array(n,1) weights: The current weights whose values have to be changed.
  :return: np.array(m,1) of the new weights.
  """
  #delw_array = np.full(len(weights[0]), delw)
  #print("delw:", delw.shape)
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
  #print("input 0:", input[0])
  weights = initialize_weights(input[0])
  #print("iniial wts", weights)
  epch = 1
  difference = 10000
  previous_error = 0
  while (-threshold > difference or difference > threshold):
    print("Epoch number:", epch)
    predicted_output = forward_pass(input, weights)
    error = calculate_error(predicted_output, actual_output)
    print("error:", error)
    delw = backward_pass(input, predicted_output, actual_output)
    weights = change_weights(delw, weights) # changing weights to the new values of weights.
    #print("weights:", weights)
    if error > 0.2 + previous_error and previous_error!=0:
      break
    difference = error - previous_error
    previous_error = error
    epch+= 1
    time.sleep(1)
  return weights # returning the most efficient weights to use later for prediction.

def predict(inputs, actual_output, type_of_normalization, weights, ):
  """
  ndarray (m,n) inputs: The predictor values.
  ndarray (m,1) outputs: Predictand value.
  str type_of_normalization: Describes the kind of normalization to be applied
                             to the inputs and outputs.
                             Possible types: min_max, z_score (to be added soon).
  :return: ndarray of predicted outputs
  """
  inputs, actual_output,terms = normalize(inputs, actual_output, type_of_normalization)
  predicted_output = forward_pass(inputs, weights)
  error = calculate_error(predicted_output, actual_output)
  print("error:", error)
  predicted_output_denormalized = denormalize(predicted_output, terms[0], terms[1], type_of_deormalization)
  return predicted_output_denormalized

if __name__ == "__main__":


  # Importing the dataset
  dataset = pd.read_csv('50_Startups.csv')
  X = dataset.iloc[:, :-2].values
  print("X shape", X.shape)
  y = dataset.iloc[:, 4].values
  y = y.reshape((len(y),1))
  X,y,[mu, sigma] =normalize(X, y, "z_score")
  print("normalized X shape:", X.shape)
  print("normalized y shape:", y.shape)

  # Encoding categorical data
  """from sklearn.preprocessing import LabelEncoder, OneHotEncoder
  labelencoder = LabelEncoder()
  X[:, 3] = labelencoder.fit_transform(X[:, 3])
  onehotencoder = OneHotEncoder(categorical_features = [3])
  X = onehotencoder.fit_transform(X).toarray()"""

  # Avoiding the Dummy Variable Trap
  #X = X[:, 1:]
  wts = epochs(X, y)
 
