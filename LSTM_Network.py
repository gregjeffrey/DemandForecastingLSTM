# Imports
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import datetime
import os

# ---------------------------- PARAMS  ------------------------------- #
EPOCHS = 1  # Number of epochs to train model for
LEARNING_RATE = 0.1  # Learning rate
HIDDEN_DIMS = 1  # Number of hidden dims
NUM_LSTM_CELLS = 1  # Number of LSTM cells in network
NUM_WINDOWS = 5  # Number of windows / epoch
WINDOW_LENGTH = 8760  # Number of hours for window (8760 = 1 year)
FORECAST_HORIZON = 24  # Desired predictions in future
NORMALIZATION = 'min-max'  # Normalization is min-max (if anything else then Gaussian)
DEFAULT_SAVE = [True, False, True, True]
CURRENT_EPOCH = 0  # Keep track of current epoch if training is paused


# -------------------------- Functions ----------------------------- #
def save(model, outputs, losses, test_losses=None, save=DEFAULT_SAVE):
    """
    Saves desired parameters.

    model: pytorch model

    outputs: PyTorch tensor
        Outputs of model.

    losses: NumPy array
        Array of MSE values for training epoch.

    save: [save_model, save_outputs, save_losses]
        Indicate which parameters to save
    """

    # Generate date tag and path for outputs
    time = datetime.datetime.now()
    date_tag = '{0}{1}_{2}{3}'.format(time.month, time.day, time.hour, time.minute)
    preds_path = os.getcwd() + '/predictions/{}.csv'.format(date_tag)
    model_path = os.getcwd() + '/models/model_{}.pkl'.format(date_tag)
    model_dict_path = os.getcwd() + '/models/model_{}_state_dict.pkl'.format(date_tag)
    loss_path = os.getcwd() + '/losses/loss_{}.csv'.format(date_tag)
    test_loss_path = os.getcwd() + '/losses/test_loss_{}.csv'.format(date_tag)

    # Save outputs
    if save[0]:
        torch.save(model, model_path)
        torch.save(model.state_dict(), model_dict_path.format(date_tag))

    if save[1] and outputs:
        pd.DataFrame(outputs.cpu().data.numpy()).to_csv(preds_path)

    if save[2] and losses:
        pd.DataFrame(losses).to_csv(loss_path)

    if save[3] and test_losses:
        pd.DataFrame(test_losses).to_csv(test_loss_path)


def train(lstm_model, epochs, training_inputs, training_targets,
          window_length=WINDOW_LENGTH, num_windows=NUM_WINDOWS, forecast=FORECAST_HORIZON,
          test_inputs=None, test_targets=None, lr=LEARNING_RATE):
    """
    Trains our LSTM NN model.

    Params
    ------
    lstm_model: PyTorch model
        The Neural network model to be trained.

    epochs: int
        Number of epochs to train model for.

    training_inputs: PyTorch tensor
        Inputs to train the model on.

    training_targets: PyTorch tensor
        Target values for the model to use for loss calculation.

    window_length: int
        Number of hours for window length. (8670 = 1 year)

    num_windows: int
        Number of windows per epoch.

    test_inputs: torch Tensor (default=None)
        If provided, will be used to calculate test error.

    test_targets: torch Tensor (default=None)
        If provided, will be used to calculate test error.

    lr: float
        Learning rate for model.

    Returns
    -------
    """

    global CURRENT_EPOCH  # Keep track of model total epochs

    # Initialize PyTorch criterion and optimizer functions
    criterion = nn.MSELoss()
    optimizer = optim.SGD(lstm_model.parameters(), lr=lr)

    # Initialize arrays to keep track of losses
    losses = []
    test_losses = []
    test_info = bool(test_targets and test_inputs)

    # Train loop
    for epoch in range(epochs):
        window_mses = []
        test_mses = []

        # Iterate over number of windows in an epoch
        for i in range(num_windows):
            # Randomly choose windows of desired length for training
            index = round(random.random()*(len(training_inputs)-window_length-forecast))
            window = training_inputs[index:index+window_length].unsqueeze(1)
            target = training_targets[index+forecast:index+window_length+forecast]

            # Zero gradients
            optimizer.zero_grad()

            # Calculate error and update weights
            outputs = lstm_model(window)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            window_mses.append(loss.data[0])

            # Track test error
            if test_info:
                index = round(random.random()*(len(test_inputs)-window_length-forecast))
                window = test_inputs[index:index+window_length].unsqueeze(1)
                targets = test_targets[index+forecast:index+window_length+forecast]
                test_output = lstm_model(window)
                test_mses.append(criterion(test_output, targets))

        # Append epoch error to losses array and print status
        mse = np.mean(window_mses)
        losses.append(mse)
        if test_info:
            test_mse = np.mean(test_mses)
            test_losses.append(test_mse)
            print('Epoch: {0}/{1},Training Loss: {2}, Test Loss: {3}'.format(CURRENT_EPOCH, EPOCHS, mse, test_mse))
        else:
            print('Epoch: {0}/{1},Training Loss: {2}'.format(CURRENT_EPOCH, EPOCHS, mse))

        CURRENT_EPOCH += 1  # Append global CURRENT_EPOCH counter

    # Return
    if test_info:
        return losses, test_losses
    else:
        return losses


# ------------------------ Define Network  --------------------------- #
class LSTM(torch.nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions, num_lstm_cells=1, lstm_dropout=0.1):
        """
        Initialize LSTM Neural Network.

        Params
        ------
        input_dimensions: int
            Number of inputs (window length)

        hidden_dimensions: int
            Width of LSTM and Linear hidden layers in network.

        num_lstm_cells: int (default=1)
            Number of LSTM cells in network.

        lstm_dropout: float (default = 0.1) (<1)
            Proportion of LSTM neurons that drop out during training.
        """
        super(LSTM, self).__init__()

        # Parameters
        self.input_dim = input_dimensions
        self.hidden_dim = hidden_dimensions
        self.num_lstm_cells = num_lstm_cells
        self.lstm_dropout = lstm_dropout

        # Network layers
        self.lstm = nn.LSTM(input_dimensions, hidden_dimensions, dropout=0.1, num_layers=num_lstm_cells)
        self.c1 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.out = nn.Linear(hidden_dimensions, 1, bias=False)

    def forward(self, x):
        """
        Perform forward pass of network.

        Params
        ------
        x: pytorch Tensor
            Input data to network

        Returns
        -------
        output: Tensor
            Network output.
        """
        h_1, c_1 = self.lstm(x)
        output = self.c1(h_1.squeeze(1))
        output = self.out(output)
        return output
