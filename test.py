import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

df = pd.read_csv('full_data.csv', index_col=0)
cols = ['humidity', 'precipAccumulation', 'precipIntensity', 'pressure', 'temperature', 'windSpeed', 'MWh']
df = df[cols].dropna()
y = np.array(df.MWh.values)
x = np.array(df.drop('MWh', axis=1).values)

input = Variable(torch.from_numpy(x))
target = Variable(torch.from_numpy(y))

class Sequence(nn.Module):
	def __init__(self):
		super(Sequence, self).__init__()
		self.lstm1 = nn.LSTMCell(7, 100)
		self.lstm2 = nn.LSTMCell(100,100)
		self.linear = nn.Linear(100,1)
	
	def forward(self, input, future=0):
		outputs = []
		h_t = Variable(torch.zeros(input.size(0), 100).double(), requires_grad=False)
		c_t = Variable(torch.zeros(input.size(0), 100).double(), requires_grad=False)
		h_t2 = Variable(torch.zeros(input.size(0), 100).double(), requires_grad=False)
		c_t2 = Variable(torch.zeros(input.size(0), 100).double(), requires_grad=False)
		for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
			h_t, c_t = self.lstm1(input_t, (h_t, c_t))
			h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
			output = self.linear(h_t2)
			outputs += [output]
		for i in range(future):# if we should predict the future
			h_t, c_t = self.lstm1(output, (h_t, c_t))
			h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
			output = self.linear(h_t2)
			outputs += [output]	
		# outputs = torch.stack(outputs, 1).squeeze(2)
		return outputs

np.random.seed(0)
torch.manual_seed(0)

seq = Sequence()
seq.double()
criterion = nn.MSELoss()
optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

for i in range(15):
	print('STEP: {}'.format(i))
	def closure():
		optimizer.zero_grad()
		out = seq(input)
		loss = criterion(out, target)
		print('loss:', loss.data.numpy()[0])
		loss.backward()
		return loss
	optimizer.step(closure)
