from autogluon.tabular import TabularDataset, TabularPredictor
import pickle
import os
from twilio.rest import Client

account_sid = "AC302209a332b7d9e3f441cdd0a5569ccf"
auth_token = "b467d59d57785221779f11f1400f3a37"
client = Client(account_sid, auth_token)

# time limit for training is 10 mins, 1, 4, 8, 12 hours
time_limit = [60*60, 60*60*4, 60*60*8, 60*60*12]

best_rmse = 100

for time in time_limit:

    # Load the training data
    train_data = TabularDataset('train_data2.csv')   

    # Create a predictor
    predictor = TabularPredictor(label='SOC').fit(
        train_data, 
        time_limit=time, 
        presets=['best_quality']
    )

    # load the test data
    test_data = TabularDataset('test_data.csv')

    # make predictions on the test data
    y_pred = predictor.predict(test_data)

    predictor.evaluate(test_data, silent=True)

    rmse = predictor.evaluate_predictions(y_true=test_data['SOC'], y_pred=y_pred, auxiliary_metrics=True)
    rmse = rmse['root_mean_squared_error']

    model_saved = False

    if rmse < best_rmse:
        #save the top model a pickle file
        with open('top_model2.pkl', 'wb') as f:
            pickle.dump(predictor, f)

        model_saved = True

    message = client.messages.create(
    body="Your model2 has finished training in " + str(time/60/60) + " hours. The RMSE is " + str(rmse) + ". The model was saved: " + str(model_saved),
    from_="+13613457812",
    to="+13653661086"
    )
    print(message.sid)




