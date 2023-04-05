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
    #time_limit = 10*60,
    presets='best_quality',
    verbosity=2
)

# import test data from Data/phil_socdata_test.csv, normalize (between 0 and 1) and standardize
test_data1 = TabularDataset(f'Data/phil_socdata_test1.csv')
test_data2 = TabularDataset(f'Data/phil_socdata_test2.csv')

y_pred1= predictor.predict(test_data1.drop(columns=[label]))

y_pred2= predictor.predict(test_data2.drop(columns=[label]))

#plots the predicted vs actual values of the top performing model using matplotlib
plt.plot(y_pred1, label="Predictions")
plt.plot(test_data1[label], label="True Values")
plt.legend()
plt.savefig('Data/phil_socdata_test1.png')

plt.plot(y_pred2, label="Predictions")
plt.plot(test_data2[label], label="True Values")
plt.legend()
plt.savefig('Data/phil_socdata_test2.png')

mse_test = np.mean(((y_pred1 - test_data1[label])**2))
rmse_test = math.sqrt(mse_test)
print("Test data 1 RMSE", rmse_test)
mse_test = np.mean(((y_pred2 - test_data2[label])**2))
rmse_test = math.sqrt(mse_test)
print("Test data 2 RMSE", rmse_test)


