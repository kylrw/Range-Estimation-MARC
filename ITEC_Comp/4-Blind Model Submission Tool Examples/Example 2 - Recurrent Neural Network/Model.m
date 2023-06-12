%LSTM Neural Network SOC Estimator - McMaster University 2022
function [Y_est] = Model(X)

%Normalize input data X to normalization used for trained network
MAX =   [15,    4.5,    51 ];
MIN =   [-19,   2.5,    -27];
X(:,1) = ((X(:,1) - MIN(1))./(MAX(1)-MIN(1)));
X(:,2) = ((X(:,2) - MIN(2))./(MAX(2)-MIN(2)));
X(:,3) = ((X(:,3) - MIN(3))./(MAX(3)-MIN(3)));

%Reorder and transpose X data to match neural network format
X_reordered = [X(:,2), X(:,1), X(:,3)]';

%Load trained network parameters
load("Trained_LSTM_Network_Parameters.mat");

%Estimate SOC
[updatedNet,Pr] = predictAndUpdateState(NETS{1,1},X_reordered(:,1:100));
Y_est = predict(updatedNet, X_reordered(:,:))';
end