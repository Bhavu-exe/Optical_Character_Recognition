import numpy as np
import math
import json


class OCRNeuralNetwork:
    NN_FILE_PATH = 'path_to_nn_file.json'
    LEARNING_RATE = 0.1

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, train_indices, use_file):
        self.num_hidden_nodes = num_hidden_nodes
        self.data_matrix = data_matrix
        self.data_labels = data_labels
        self.train_indices = train_indices
        self._use_file = use_file
        self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
        self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
        self.hidden_layer_bias = self._rand_initialize_weights(1, 10)

    def _rand_initialize_weights(self, size_in, size_out):
        return np.random.rand(size_out, size_in) * 0.12 - 0.06

    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def train(self, data):
        y1 = np.dot(np.mat(self.theta1), np.mat(data['y0']).T)
        sum1 = y1 + np.mat(self.input_layer_bias)  # Add the bias
        y1 = self.sigmoid(sum1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)  # Add the bias
        y2 = self.sigmoid(y2)

        actual_vals = [0] * 10
        actual_vals[data['label']] = 1
        output_errors = np.mat(actual_vals).T - np.mat(y2)
        hidden_errors = np.multiply(np.dot(np.mat(self.theta2).T, output_errors), self.sigmoid_prime(sum1))

        self.theta1 += self.LEARNING_RATE * np.dot(np.mat(hidden_errors), np.mat(data['y0']))
        self.theta2 += self.LEARNING_RATE * np.dot(np.mat(output_errors), np.mat(y1).T)
        self.hidden_layer_bias += self.LEARNING_RATE * output_errors
        self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(np.mat(self.theta1), np.mat(test).T)
        y1 = y1 + np.mat(self.input_layer_bias)  # Add the bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)  # Add the bias
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": [np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
            "b1": self.input_layer_bias[0].tolist()[0],
            "b2": self.hidden_layer_bias[0].tolist()[0]
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])]