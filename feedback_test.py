
# coding: utf-8

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
import matplotlib


# ---------------------------- PARAMS  ------------------------------- #
EPOCHS = 1
LEARNING_RATE = 0.1
HIDDEN_DIMS = 1
NUM_LSTM_CELLS = 1
NUM_WINDOWS = 5
WINDOW_LENGTH = 8760 #1-year windows
FORECAST_HORIZON = 1
MODEL_PATH = '.\models\model1hr100epoch.pkl'


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
        output = self.c1(h_1.squeeze(1)).cuda()
        output = self.out(output)
        return output.cuda()



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

#Splits timestamps for plotting later
(training_t,test_t) = np.split(data['index'].values,[num_training_samples])

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
model = LSTM(input_dim, HIDDEN_DIMS, num_lstm_cells=NUM_LSTM_CELLS).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

model = torch.load(MODEL_PATH,map_location=lambda storage, loc: storage).cuda()

def test_model():
    #Apply test data
    index = round(random.random()*(len(test_inputs)-WINDOW_LENGTH-FORECAST_HORIZON)) #Selects a random index from test data

    t = test_t[index+FORECAST_HORIZON:index+WINDOW_LENGTH+FORECAST_HORIZON] #timestamps for plotting
    t = matplotlib.dates.datestr2num(t)

    window = test_inputs[index:index+WINDOW_LENGTH].unsqueeze(1)
    target = test_targets[index+FORECAST_HORIZON:index+WINDOW_LENGTH+FORECAST_HORIZON]
    output = model(window)
    
    return output,target,t


def plot_outputs(output,target,t,frame):#De-normalize outputs and targets for plotting
    op = output.squeeze(1).data.numpy()
    tg = target.data.numpy()
    mx = data['MWh'].max()
    mn = data['MWh'].min()
    op = op*(mx-mn) + mn
    tg = tg*(mx-mn) + mn
    frame = -1*frame
    
    x = matplotlib.dates.num2date(t[frame:])
    y1 = op[frame:]
    y2 = tg[frame:]

    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.xlabel('Date/Hour')
    plt.ylabel('Demand (MWh)')
    plt.title('Predicted vs. Actual Demand: Day-Ahead')
    plt.xticks(rotation=45)


def test_feedback(n=1):
    
    index = round(random.random()*(len(test_inputs)-WINDOW_LENGTH-FORECAST_HORIZON)) #Selects a random index from test data
    window = test_inputs[index:index+WINDOW_LENGTH].unsqueeze(1)
    
    #Perform multi-step prediction using feedback loop
    for i in range(n):
        output = model(window)
        last_hr_output = output[-1].data[0] #Saves last hour output
        #Access the future weather data
        feedback = test_inputs[index:index+WINDOW_LENGTH+i+1][-1].unsqueeze(1).data
        feedback[2] = last_hr_output #Replace power data with network output
        #Shift the input window 1 step in the future and append the last output
        window = Variable(torch.cat((window[1:].data,feedback.view(1,1,3)),0)).cuda()

    preds = output[-24:].data.numpy()
    actuals = test_targets[index+FORECAST_HORIZON+24:index+WINDOW_LENGTH+FORECAST_HORIZON+24][-24:].data.numpy()

    return preds
    return actuals
            #feedback_data =  #Replace the known power data with the network output
            #window = window[1:].append(feedback_data) #Append the feedback data to the window




