# SOCprediction
Initial Project for MARC

use https://data.mendeley.com/datasets/cp3473x7xv/3 dataset

Use first 3 rows (voltage, current, temperature) to use as input for LSTM model to predict SOC

only use the first 100 000 data points

use 10 hidden units

1000 epochs

Notes:
-To create the model, I just copied your general design from the MATLAB file.
-The predictions aren't as accurate when the SOC has a negative slope.
-The prediction is a bit different everytime I run it.
-Sometimes it will just be a horizontal line, if this happens I have a while loop so the model gets trained again
-1000 epochs didn't seem to have much more of an effect compared to 100, I left it at 100 so its easier to debug but it can be easily changed

