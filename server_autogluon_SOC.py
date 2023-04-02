from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import math
import matplotlib.pyplot as plt

# Import and preview training data
train_data = TabularDataset('Data/phil_socdata_train.csv')
train_data.head()
label = 'SOC'

# Model training with AutoGluon
predictor = TabularPredictor(label=label).fit(
    train_data,
    presets='best_quality',
    auto_stack=True,
    ag_args_fit={'num_gpus': 1}
)

# Import and preview test data
test_data = TabularDataset(f'Data/phil_socdata_test.csv')

# Make predictions and evaluate the model
y_pred = predictor.predict(test_data.drop(columns=[label]))
predictor.evaluate(test_data, silent=True)
predictor.leaderboard(test_data, silent=True)

# Plot the predicted vs actual values
plt.plot(y_pred, label="Predictions")
plt.plot(test_data[label], label="True Values")
plt.legend()
plt.savefig("Pictures/predictions2.png")

# Smooth the predicted values using a moving average of 600 values and plot them against the actual values
y_pred_smooth = y_pred.rolling(600).mean()
plt.plot(y_pred_smooth, label="Predictions")
plt.plot(test_data[label], label="True Values")
plt.legend()
plt.savefig("Pictures/smoothed_predictions.png")

# Calculate the RMSE for both the original and smoothed predictions
mse_test = np.mean(((y_pred - test_data[label])**2))
rmse_test = math.sqrt(mse_test)
print("test data rmse", rmse_test)

mse_test_smooth = np.mean(((y_pred_smooth - test_data[label])**2))
rmse_test_smooth = math.sqrt(mse_test_smooth)
print("test data rmse (smooth)", rmse_test_smooth)

# Calculate the accuracy of the original and smoothed predictions
accuracy = sum(abs(y_pred - test_data[label]) < 0.1) / len(y_pred)
accuracy_smooth = sum(abs(y_pred_smooth - test_data[label]) < 0.1) / len(y_pred_smooth)

print("accuracy", accuracy)
print("accuracy_smooth", accuracy_smooth)