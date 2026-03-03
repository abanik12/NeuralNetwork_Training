import numpy as np
import gzip
import pickle

def load_data():
    """
    Load the MNIST dataset from a gzipped pickle file.

    Returns:
        tuple: A tuple containing training_data, validation_data, and test_data.
    """
    # Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/
    # and save it as 'mnist.pkl.gz' in the same directory as this script.
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    Load the MNIST dataset and preprocess it for use in the neural network.

    Returns:
        tuple: A tuple containing training_data, validation_data, and test_data.
               Each dataset is formatted as a list of tuples (x, y), where:
               - x is a 784-dimensional NumPy array representing the input image.
               - y is the corresponding label (one-hot encoded for training data).
    """
    training_data, validation_data, test_data = load_data()

    # Preprocess training data
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]  # Flatten images
    training_results = [vectorized_result(y) for y in training_data[1]]    # One-hot encode labels
    training_data = list(zip(training_inputs, training_results))

    # Preprocess validation data
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = list(zip(validation_inputs, validation_data[1]))

    # Preprocess test data
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return training_data, validation_data, test_data


def vectorized_result(j):
    """
    Convert a digit (0-9) into a one-hot encoded vector.

    Args:
        j (int): The digit to be converted.

    Returns:
        np.ndarray: A 10-dimensional one-hot encoded vector.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e