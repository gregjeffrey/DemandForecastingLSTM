
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import matplotlib.pyplot as plt


# In[18]:


data = pd.read_csv('full_data.csv',delimiter=',',header=0,index_col=0)

temp = data['apparentTemperature']
temp = (temp - temp.min())/(temp.max()-temp.min())
hum = data['humidity']
hum = (hum - hum.min())/(hum.max() - hum.min())
power = data['MWh']
power = (power - power.min())/(power.max() - power.min())

net_input_dict = dict({
    'temperature':temp,
    'humidity':hum,
    'MWh':power
    })

df = pd.DataFrame(net_input_dict) ##Input data is a 61392x9 matrix

net_targets = power.values #Targets are a 61392 column vector
net_inputs = df.values


# In[19]:


##Define the model
class SimpleRNN(nn.Module):
    def __init__(self, hidden_size,num_features):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.inp = nn.Linear(num_features, hidden_size) ##Input layer: fully-connected layer, takes in feature #, outputs hidden size
        
        ##Use 1 LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, dropout=0.05) 
        self.out = nn.Linear(hidden_size, 1) ##Output Layer: fully-connected layer, takes in hidden size, outputs size 1

    #def step(self, input, hidden=None):
        #Input is one step of the sequence, or 1x9
        
    def forward(self, input, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs) ##Sets the # of steps to the # of inputs
            
        outputs = Variable(torch.zeros(steps, 1, 1)) ##Initializes the output tensor (vector of # of steps)

        #unsqueeze method adds a dimension of size 1
        #Linear layer
        #Input size: (N,1,num_features)
        #Outut size: (N,1,hidden_size)
        lstm_input = self.inp(inputs.view(-1,1,num_features)) ##Applies the data to the input layer, returns input variable
        print(lstm_input.shape)
        #LSTM Layer
        #Input size: (seq_length,batch,input_size)
        #Output size: (seq_length,batch,hidden_size)
        lstm_output, hidden = self.lstm(lstm_input, hidden) ##Applies the input variable to the rnn layer(s), returns output and hidden states
        print(lstm_output.shape)
        #Linear Layer
        #Input size: (N,1,hidden_size)
        #Output size:(N,1,1)
        output = self.out(lstm_output.squeeze(1)) ##Applies the LSTM output to the linear output layer, returns output
        print(output.shape)
        return output, hidden
    


# In[20]:


n_epochs = 1000
hidden_size = 20
num_features=net_inputs.shape[1]

model = SimpleRNN(hidden_size,num_features).cuda()
criterion = nn.MSELoss() #Sets loss function to MSE
optimizer = optim.SGD(model.parameters(), lr=0.01) 

losses = np.zeros(n_epochs) # #Initializes loss variable, for plotting

#torch.cuda.get_device_name(0)
#Network training

inputs = Variable(torch.from_numpy(net_inputs[:-1]).float()).cuda()
targets = Variable(torch.from_numpy(net_targets[1:]).float()).cuda()
 
print(torch.cuda.get_device_name(0))    


# In[ ]:


for epoch in range(n_epochs):

    outputs, hidden = model(inputs, None)
    optimizer.zero_grad() ##Clears the gradients
    loss = criterion(outputs, targets) ##Calculates loss
    loss.backward() ##Performs backpropagation
    optimizer.step()

    losses[epoch] += loss.data[0]

    print(epoch, loss.data[0])

    # Use some plotting library
    # if epoch % 10 == 0:
        # show_plot('inputs', _inputs, True)
        # show_plot('outputs', outputs.data.view(-1), True)
        # show_plot('losses', losses[:epoch] / n_iters)

        # Generate a test
        # outputs, hidden = model(inputs, False, 50)
        # show_plot('generated', outputs.data.view(-1), True)



# In[3]:




