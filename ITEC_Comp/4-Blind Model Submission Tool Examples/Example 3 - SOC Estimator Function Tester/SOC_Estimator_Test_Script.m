%SOC Estimator Function Tester
%McMaster University 2022

%Run this script to determine if an SOC estimator function is working
%properly
clear all; close all;
current_folder = pwd; 

%Select and load measured data
[file,path] = uigetfile('Battery_Data.mat','Select data file');
load(fullfile(path, file)); 

%Declare matrix of measured input data X
X = [meas.Current meas.Voltage meas.Battery_Temp_degC];
%Declare array of measured output data Y
Y = meas.SOC;

%Select folder with Model
cd(uigetdir('','Select folder which includes Blind_Model.zip'));
mkdir 'Blind Model'
unzip('Blind_Model.zip','Blind Model');
cd('Blind Model')

%Open settings file, display read email
[~,Author_Email]=xlsread('Settings.xlsx','B3:B3')
[~,Model_Name]=xlsread('Settings.xlsx','B4:B4')

%Calculate estimate SOC, Y_est, using model
Y_est = Model(X);

%Calculator error 
RMSE    =   100*sqrt(mean((Y(:)-Y_est(:)).^2)); %RMS error in percent
MAE     =   100*(mean(abs(Y(:)-Y_est(:)))); %Mean absolute error in percent
MAXE    =   100*max(abs(Y(:)-Y_est(:))); %Max error in percent

%Plot actual versus estimated SOC and error
figure
subplot(2,1,1) %Plot actual SOC (Y) versus estimated SOC (Y_est)
    plot([1:1:length(Y)]./3600,(Y.*100))
    hold on
    plot([1:1:length(Y)]./3600,Y_est.*100)
    ylim([0 100])
    ylabel('SOC (%)')
    xlabel('Time (Hour)')
    legend('Actual','Estimated')
    grid on
subplot(2,1,2)
    plot([1:1:length(Y)]./3600,(Y-Y_est).*100)
    legend(['RMSE ' num2str(RMSE,2) ' %'])
    ylabel('SOC Estimation Error (%)')
    xlabel('Time (Hour)')
    grid on

cd ../ %move up one folder level
rmdir('Blind Model','s'); %Remove temporary blind model directory
cd(current_folder);  %Return to original directory