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
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    source = request.form['source']
    dest = request.form['dest']

    # get the api key from the login_creds.py file
    api_key = os.getenv('api_key')
    gmaps = googlemaps.Client(key=api_key)

    # Request directions via driving
    now = datetime.now()
    directions_result = gmaps.directions(source,
                                        dest,
                                        mode="driving",
                                        departure_time=now)

    # print the distance between the two points
    distance = directions_result[0]['legs'][0]['distance']['text']

    # appends the start and end locations of each step to a list
    locations = []
    for i in range(len(directions_result[0]['legs'][0]['steps'])):
        locations.append([directions_result[0]['legs'][0]['steps'][i]['start_location']['lat'],directions_result[0]['legs'][0]['steps'][i]['end_location']['lng']])

    # Request elevation data
    elevations = gmaps.elevation(locations)

    # appends the distance and time of each step to a list
    step_time = []
    step_distance = []
    for i in range(len(directions_result[0]['legs'][0]['steps'])):
        step_time.append(directions_result[0]['legs'][0]['steps'][i]['duration']['text'])
        step_distance.append(directions_result[0]['legs'][0]['steps'][i]['distance']['text'])

    # iterate through the list and convert the time and distance to hours and km
    for i in range(len(step_time)):
        # split at the space
        step_time[i] = step_time[i].split()
        step_distance[i] = step_distance[i].split()

        if step_time[i][1] == "mins" or step_time[i][1] == "min":
            step_time[i][0] = float(step_time[i][0]) / 60
        if step_distance[i][1] == "m":
            step_distance[i][0] = float(step_distance[i][0]) / 1000

    #for every step, remove the string and leave only the number
    for i in range(len(step_time)):
        step_time[i] = step_time[i][0]
        step_distance[i] = step_distance[i][0]

    avg_speed = 0

    # print the average speed of every step
    for i in range(len(step_time)):
        avg_speed += float(step_distance[i]) / float(step_time[i])

    avg_speed = avg_speed / len(step_time)

    return render_template('index.html', source=source, dest=dest, time=directions_result[0]['legs'][0]['duration']['text'],
                       avg_speed=round(avg_speed, 2),
                       distance=distance,
                       elevation_change=round(elevations[0]['elevation'] - elevations[-1]['elevation'], 2))

if __name__ == '__main__':
    app.run(debug=True)

