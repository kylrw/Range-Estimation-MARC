"""

This program will use 4 google maps API,s to get:
-the distance between two points
-the elevation change between two points
-the time it takes to drive between two points
-the average driving speed between two points

"""

import os
import requests
import googlemaps
import login_creds
from datetime import datetime

api_key = os.getenv('api_key')
gmaps = googlemaps.Client(key=api_key)


# Take source as input
source = "57 Joanne Crt, Ancaster ON Canada"
#input("Enter source: ")
  

# Take destination as input
dest = "145 Bowmen St, Hamilton ON Canada"


# Request directions via driving
now = datetime.now()
directions_result = gmaps.directions(source,
                                     dest,
                                     mode="driving",
                                     departure_time=now)


# print the directiions step by step
#for i in range(len(directions_result[0]['legs'][0]['steps'])):
    #for direction in directions_result[0]['legs'][0]['steps'][i]['html_instructions']:
       # print(direction, end="")
   # print()

# print the distance between the two points
distance = directions_result[0]['legs'][0]['distance']['text']

# appends the start and end locations of each step to a list
locations = []
for i in range(len(directions_result[0]['legs'][0]['steps'])):
    locations.append([directions_result[0]['legs'][0]['steps'][i]['start_location']['lat'],directions_result[0]['legs'][0]['steps'][i]['end_location']['lng']])

# Request elevation data
elevations = gmaps.elevation(locations)

# Request the average speed of every step in locations
speeds = gmaps.snapped_speed_limits(locations)

# print the average speed of every step
print(speeds)

