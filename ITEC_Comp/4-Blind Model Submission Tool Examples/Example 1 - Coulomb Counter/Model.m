%Coulomb Counting SOC Estimator - McMaster University 2022
function [Y_est] = Model(X)

%Input X: Measured current, voltage, and temperature values
%X: 3 columns, T rows, where T is length of input data in seconds
Current = X(:,1); %Amps, column 1
%Current: negative-discharging, positive-charging
Voltage = X(:,2); %Volts, column 2
Temperature = X(:,3); %degrees Celsius, column 3
Time = (0:1:(length(Current)-1))'; %seconds

%Coulomb Counting SOC Estimator
SOC_Init = 1; %Assume battery always starts fully charged
Capacity = 4.6; %Ah, Nominal capacity of new Tesla 21700 NMC/NCA cell

%Coulomb counting: SOC = integral of current
for i=1:length(Time)
    if i==1
        %At time step 0, SOC is equal to initial setpoint
        SOC(i)=SOC_Init;
    else
        %Greater than time step 0
        %SOC = SOC of last time step + delta SOC
        SOC(i)=SOC(i-1)+Current(i)*((Time(i)-Time(i-1))/3600)/Capacity;
    end
end

%Output Y: Estimated SOC
%Y: 1 columns, T rows, where T is length of input data in seconds
Y_est=SOC'; %Transpose SOC from columns to rows  
end