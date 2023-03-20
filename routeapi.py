"""

This program will use 4 google maps API,s to get:
-the distance between two points
-the elevation change between two points
-the time it takes to drive between two points
-the average driving speed between two points

"""

import urllib.request
import os
import requests
import json
import time
import datetime

import login_creds

api_key = os.getenv('api_key')

# Take source as input
source = "Ancaster"
#input("Enter source: ")
  
# Take destination as input
dest = "NiagaraFalls"
#input("Enter destination:")
  
# url variable store url 
url ='https://maps.googleapis.com/maps/api/distancematrix/json?'
  
# Get method of requests module
# return response object
r = requests.get(url + 'origins=' + source +
                   '&destinations=' + dest +
                   '&key=' + api_key)
                     
# json method of response object
# return json format result
x = r.json()
  
# by default driving mode considered
  
# print the value of x
print(x)