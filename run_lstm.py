import pandas as pd
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np

# ---------------------------- PARAMS  ------------------------------- #
EPOCHS = 1000


# ------------------------ Define Network  --------------------------- #
class LSTM(torch.nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions):
        super(LSTM, self).__init__()

        # Parameters
        self.input_dim = input_dimensions
        self.hidden_dim = hidden_dimensions

        # Network layers
        self.lstm = nn.LSTM(input_dimensions, hidden_dimensions, dropout=0.1)
        self.lstm2 = nn.LSTM(hidden_dimensions, hidden_dimensions, dropout=0.1)
        self.fullycn = nn.fcn(hidden_dimensions, 50)
        self.out = nn.Linear(input_dimensions, 1, bias=False)

    def forward(self, x):
        h_1, c_1 = self.lstm(x)
        h_2, c_2 = self.lstm(h_1)
        output = self.fullycn(h_2.squeeze(1))
        output = self.out(output)
        return output


# ---------------------- Load and Process Data  ---------------------- #
df = pd.read_csv('full_data.csv', index_col=0)
cols = ['apparentTemperature', 'dewPoint', 'humidity',
        'pressure', 'temperature', 'MWh']

df = df[cols]
df['next_pow'] = df.shift(-1).MWh

# Obtain target data
y = np.array(df.next_pow.values)
target = Variable(torch.from_numpy(y)).float()

# Normalize input data
df = (df - df.mean())/df.std()
x = np.array(df.drop('next_pow', axis=1).values)
inputs = Variable(torch.from_numpy(x)).float()
inputs = inputs.unsqueeze(1)

# -------------------- Instantiate LSTM Network  --------------------- #
# Model Params
input_dim = inputs.shape[2]
hidden_dim = 75

# Create model and necessary functions
model = LSTM(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.10)


# --------------------------- Train Network -------------------------= #
losses = []

# Train loop
for epoch in range(EPOCHS):
    # Zero gradients
    optimizer.zero_grad()

    # Update weights
    outputs = model(inputs)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    # Print stats
    mse = loss.data[0]
    losses.append(mse)
    print('Epoch: {0}/{1}, Loss: {2}'.format(epoch, EPOCHS, mse))


# Generate date tag
time = datetime.datetime.now()
date_tag = '{0}{1}_{2}{3}'.format(time.month, time.day, time.hour, time.minute)
preds_path = os.getcwd() + '\predictions\{}.csv'.format(date_tag)
model_path = os.getcwd() + '\models\model_{}.pkl'.format(date_tag)
model_dict_path = os.getcwd() + '\models\model_{}_state_dict.pkl'.format(date_tag)

# Save
torch.save(model, model_path)
torch.save(model.state_dict(), model_dict_path.format(date_tag))
pd.DataFrame(outputs.cpu().data.numpy()).to_csv(preds_path)
