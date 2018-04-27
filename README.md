# MLProject
Electricity Demand Forecasting using LSTM Neural Networks

This project was done by Greg Jeffrey and Naveen Sehgal for completion of Northeastern University's EECE5644 - Intro Machine Learning and Pattern Recognition. 

An implementation of a Long Short-Term Memory neural network for the task of predicting electricity demand is included in this repository.

The neural network model, defined in LSTM_Network.py, can be trained using train_lstm_model.py and given custom parameters.

Data used in the model can be extracted from ISO-New England and from DarkSky API using get_isone_data.py and get_temp_data.py, respectively (note: for DarkSky, a paid API account is required.) For convenience, a 7-year data set is included in full_data.csv.


