# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os

class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25
        self.nuOutput = 1

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        np.random.seed(1)
        self.nuInputNodes = input_dim + 1 #including bias
        self.hidden_layer_bool = hidden_layer

        if self.hidden_layer_bool:
            self.nuNeurons = [self.hidden_units,self.nuOutput]
        else:
            self.nuNeurons = [self.nuOutput]
        self.nuLayers = len(self.nuNeurons)

        self.weights = []
        
        previousSize = self.nuInputNodes
        for size in self.nuNeurons:
            w_shape = (previousSize, size)
            #print("Initializing weight to shape:", w_shape)
        
            weight = np.random.normal(0,1/np.sqrt(self.nuInputNodes), w_shape)
            self.weights.append(weight)
            previousSize = size


        self.grads = [None for i in range(self.nuLayers)]
        self.layer_outputs = [None for i in range(self.nuLayers)]
        self.weighted_sums = [None for i in range(self.nuLayers)]



    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_dot(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def backward(self, X, output, target):
        # Executing backward step

        self.grads[-1] = np.array(self.sigmoid_dot(self.weighted_sums[-1])*(target-output))

        if self.hidden_layer_bool:
            z = self.sigmoid_dot(self.weighted_sums[1])
            self.grads[0] = np.array((self.grads[1] @ self.weights[1].T) * z)

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""

        # Adding bias
        self.x_train = np.insert(self.x_train, self.x_train.shape[1], 1, axis=1)
        self.x_test = np.insert(self.x_test, self.x_test.shape[1], 1, axis=1)

        print("Started training with hidden layer" if self.hidden_layer_bool else "Started training without hidden layer(perceptron)")


        for epoch in range(self.epochs):
            if epoch % 50 == 0:
                print(f"Completed {epoch}/{self.epochs} epochs")
            for x,y in zip(self.x_train, self.y_train):
                y_predict = self.predict(x)
                self.backward(x, y_predict, y)

                for layer in range(self.nuLayers):
                    self.weights[layer] += np.array(self.lr * (self.layer_outputs[layer] * self.grads[layer]))

        print("Finished training with hidden layer\n" if self.hidden_layer_bool else "Finished training without hidden layer(perceptron)\n")
                
        

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        x = x.reshape((self.nuInputNodes, 1))
        self.layer_outputs[0] = x

        for n in range(self.nuLayers-1):
            ws = self.layer_outputs[n].T @ self.weights[n]
            self.weighted_sums[n] = ws
            self.layer_outputs[n+1] = self.sigmoid(ws[0]).reshape((25,1))

        ws_last = self.layer_outputs[-1].T @ self.weights[-1]
        self.weighted_sums[-1] = ws_last
        return self.sigmoid(ws_last)


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
