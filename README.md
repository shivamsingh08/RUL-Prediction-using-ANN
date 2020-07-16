# RUL-Prediction-using-ANN
Problem Statement:

Drilling process, one of the most commonly used machining processes, is selected here as a test-bed. The objective is to predict the Remaining Useful Life (RUL) of drill-bit
during the machining process by utilizing thrust-force and torque signals captured by a dynamometer during the drilling cycle (constituting a logical sensor signal segment). Tests
were conducted on HAAS VF-1 CNC Machining Center with Kistler 9257B piezo-dynamometer (sampled at 250Hz) to drill holes in ¼ inch stainless steel bars. High-speed twist drill-bits
with two flutes were operated at feed rate of 4.5 inch/minute and spindle-speed at 800 rpm without coolant. Each drill-bit was used until it reached a state of physical failure
either due to excessive wear or due to gross plastic deformation of the tool tip due to excessive temperature (resulting from excessive wear).
The objective of the project is to build a model from the data which predicts the RUL by optimizing the performance evaluated in the form of Median RMSEfor testing dataset made up
of the last five holes for each drill bit.



Using GA for tuning Hyperparameters for the Neural Network:

- Use the link https://github.com/subpath/neuro-evolution.git to get the pre-built-in
package for hyperparameter tuning.
Best parameters come out to be:
"epochs": 20,
"batch_size": 40,
"n_layers": 2,
"n_neurons": 60,
"dropout": 0.2,
"optimizers": "nadam",
"activations": "relu",
"last_layer_activations": "relu",
"losses": "mean_squared_error",
"metrics": "MSE"
