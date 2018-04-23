import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random,datetime,os
import matplotlib.pyplot as plt

# ---------------------------- PARAMS  ------------------------------- #
EPOCHS = 1
LEARNING_RATE = 0.1
HIDDEN_DIMS = 1
NUM_LSTM_CELLS = 1
NUM_WINDOWS = 5
WINDOW_LENGTH = 8760 #1-year windows
FORECAST_HORIZON = 1

# ------------------------ Define Network  --------------------------- #
class LSTM(torch.nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions, num_lstm_cells=1, lstm_dropout=0.1):
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
        h_1, c_1 = self.lstm(x)
        output = self.c1(h_1.squeeze(1))
        output = self.out(output)
        return output



# ---------------------- Load and Process Data  ---------------------- #
data = pd.read_csv('full_data.csv', index_col=0)
cols = ['apparentTemperature', 'humidity','MWh']
df = data[cols]

df = (df - df.min())/(df.max()-df.min()) ##Min-Max Normalization
#df = (df - df.mean())/df.std() ##Gaussian normalization

inputs = df
targets = df['MWh'] #Un-normalized targets

#Percentage of samples to use as training data
TRAINING_SAMPLE_RATIO = 0.7
num_training_samples = round(len(inputs)*TRAINING_SAMPLE_RATIO)

#Splits data samples 
(training_inputs,test_inputs) = np.split(inputs.values,[num_training_samples])
(training_targets,test_targets) = np.split(targets.values,[num_training_samples])

#Prepares training data for input to network
training_inputs = Variable(torch.from_numpy(training_inputs).float()).cuda()
training_targets = Variable(torch.from_numpy(training_targets).float()).cuda()
test_inputs = Variable(torch.from_numpy(test_inputs).float()).cuda()
test_targets = Variable(torch.from_numpy(test_targets).float()).cuda()


# -------------------- Instantiate LSTM Network  --------------------- #
# Model Params
input_dim = training_inputs.shape[1]
hidden_dim = HIDDEN_DIMS

# Create model and necessary functions
model = LSTM(input_dim, HIDDEN_DIMS, num_lstm_cells=NUM_LSTM_CELLS)#.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)



# -------------------- LOAD MODEL FROM FILE ----------------------------#
#Perform multi-step prediction based on trained model
def multi_step_test():
    index = round(random.random()*(len(test_inputs)-WINDOW_LENGTH-FORECAST_HORIZON))
    window = test_inputs[index:index+WINDOW_LENGTH].unsqueeze(1)
    target = test_targets[index+FORECAST_HORIZON:index+WINDOW_LENGTH+FORECAST_HORIZON]

    for i in range(24):
        output = model(window)
        last_hr_output = output[-1] #Saves last hour output
        
        #Access the future weather data
        weather_data = test_inputs[index+WINDOW_LENGTH+i+1].unsqueeze(1)
        feedback_data = weather_data[-1] #Access the new weather data
        
        feedback_data[2] = last_hr_output #Replace the known power data with the network output
        window = window[1:].append(feedback_data) #Append the feedback data to the window
        
    #The predicted outputs should now be the last 24 values of window
    pred = window[-24:]
    actual = test_targets[index+WINDOW_LENGTH:index+WINDOW_LENGTH+24]
    
    return pred,actual



