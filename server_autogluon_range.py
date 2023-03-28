from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.pyplot as plt
import numpy as np
import math

train_data = TabularDataset(f'Data/phil_rangedata_train.csv')
train_data.head()

label = 'Final SOC'
train_data[label].describe()

predictor = TabularPredictor(label=label,eval_metric='root_mean_squared_error').fit(train_data, presets='best_quality', verbosity=1, time_limit = 30) 

test_data = TabularDataset(f'Data/phil_rangedata_test.csv')

y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()

predictor.evaluate(test_data, silent=True)

#plots the predicted vs actual values of the top performing model using matplotlib
plt.plot(y_pred, label="Predictions")
plt.plot(test_data[label], label="True Values")
plt.legend()
plt.savefig("Pictures/range_predictions.png")

mse_test = np.mean(((y_pred - test_data[label])**2))
rmse_test = math.sqrt(mse_test)
print("test data rmse:", rmse_test)