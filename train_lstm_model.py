# Imports
from LSTM_Network import *
from torch.autograd import Variable

# ---------------------- Load and Process Data  ---------------------- #
data = pd.read_csv('full_data.csv', index_col=0)
cols = ['apparentTemperature', 'humidity','MWh']
df = data[cols]

if NORMALIZATION == 'min_max':
    df = (df - df.min())/(df.max()-df.min())  # Min-Max Normalization
else:
    df = (df - df.mean())/df.std()  # Gaussian normalization

inputs = df
targets = df['MWh']  # Un-normalized targets

# Percentage of samples to use as training data
TRAINING_SAMPLE_RATIO = 0.7
num_training_samples = round(len(inputs)*TRAINING_SAMPLE_RATIO)

# Splits data samples
(training_inputs, test_inputs) = np.split(inputs.values, [num_training_samples])
(training_targets, test_targets) = np.split(targets.values, [num_training_samples])

# Prepares training data for input to network
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


# --------------------------- Train Network -------------------------= #
losses, test_losses = train(model, EPOCHS, NUM_WINDOWS, LEARNING_RATE)
save(model, [], losses, test_losses)
