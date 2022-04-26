# flaskAppMultiThread.py

try:
    import asyncio
except ImportError:
    raise RuntimeError("This example requries Python3 / asyncio")

from bokeh.server.server import BaseServer
from bokeh.server.tornado import BokehTornado
from bokeh.embed import server_document
from bokeh.server.util import bind_sockets
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

from flask import Flask, render_template, request
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from threading import Thread

import holoviews as hv
import panel as pn
import xarray as xr
import json

hv.extension('bokeh')

app = Flask(__name__)

import os

path = "data/"

# creating a dict that contain all the netcdf files
datasets = {}
files = os.listdir(path)
for f in files:
    datasets[f] = xr.open_dataset('data/' + f)

# Get a list of all file names
fileNames = list(datasets.keys())


# List of function to obtain the structure of the files and dataset
def fileKeys(file):
    return list(datasets[file].keys())


def dsCoords(file, dataset):
    coords = list(datasets[file][dataset].coords)
    try:
        coords.remove('lat')
    except:
        pass
    try:
        coords.remove('lon')
    except:
        pass
    return coords


def coordsValues(file, dataset, coords):
    numOfVal = []
    for coord in coords:
        numOfVal.append(len(datasets[file][dataset].isel()[coord]))
    return numOfVal


def coordsValuesName(file, dataset, coords):
    names = []
    for coord in coords:
        names.append(datasets[file][dataset][coord].values)
    return names

def lenOfCoordValues(file, dataset, coord):
    return len(datasets[file][dataset][coord])


# Innitiating containero.js
SelectedFile = fileNames[1]
SelectedDataset = fileKeys(SelectedFile)[0]
coordsWithValue = {}
for coord in dsCoords(SelectedFile, SelectedDataset):
    coordsWithValue[coord] = 0
SelectedValues = list(coordsWithValue.values())
# Set all the data in a dict
optionData = {
    'file': SelectedFile,
    'dataset': SelectedDataset,
    'coordsAndValues': coordsWithValue
}
# writing the data in the json
json_string = json.dumps(optionData)
with open('containero.json', 'w') as outfile:
    outfile.write(json_string)


# Function creating the image
def Images(file, dataset, isels):
    xarrToPlot = datasets[file][dataset].isel(isels)
    image = hv.Image(xarrToPlot, kdims=["lon", "lat"])
    return image.opts(colorbar=True, height=500, width=800, tools=['hover'])


# Bokeh app function
def viz(doc):
    f = open("containero.json")
    selected = json.load(f)
    image = Images(selected['file'],
                   selected['dataset'],
                   selected['coordsAndValues'])
    f.close()
    model = pn.Column(image).get_root()
    doc.add_root(model)


sockets, port = bind_sockets("localhost", 0)

hvapp = Application(FunctionHandler(viz))


# locally creates a page
@app.route('/', methods=['GET', 'POST'])
def hv_page():
    # just set default selected values
    SelectedFile = fileNames[0]
    SelectedDataset = fileKeys(SelectedFile)[0]
    coordsWithValue = {}
    for coord in dsCoords(SelectedFile, SelectedDataset):
        coordsWithValue[coord] = 0
    SelectedValues = list(coordsWithValue.values())

    # update the json after submit
    if request.method == 'POST':
        #update the file selected
        SelectedFile = request.form['file']
        #check if the file changed and update the dataset selected
        if request.form['dataset'] in fileKeys(SelectedFile):
            SelectedDataset = request.form['dataset']
        else:
            SelectedDataset = fileKeys(SelectedFile)[0]
        coordsWithValue = {}
        # verification coords
        for coord in dsCoords(SelectedFile, SelectedDataset):
            try:
                #this if is here to prevent the bug where, if you select a time with a value too high
                #it can cause problem when selecting a dataset with the same name coord name but not the same length
                if int(request.form[coord]) >= lenOfCoordValues(SelectedFile, SelectedDataset, coord):
                    coordsWithValue[coord] = 0
                else:
                    coordsWithValue[coord] = int(request.form[coord])
            except:
                coordsWithValue[coord] = 0
        SelectedValues = list(coordsWithValue.values())
        optionData = {
            'file': SelectedFile,
            'dataset': SelectedDataset,
            'coordsAndValues': coordsWithValue
        }
        json_string = json.dumps(optionData)
        with open('containero.json', 'w') as outfile:
            outfile.write(json_string)

    #script containing the app
    script = server_document('http://localhost:%d/hvapp' % port)
    return render_template("index.html", script=script, template="Flask",
                           files=fileNames, savedFileOpt=SelectedFile,
                           dataset=fileKeys(SelectedFile), savedDsOpt=SelectedDataset,
                           coords=dsCoords(SelectedFile, SelectedDataset),
                           values=coordsValues(SelectedFile, SelectedDataset, dsCoords(SelectedFile, SelectedDataset)), savedValuesOpt=SelectedValues,
                           valuesName=coordsValuesName(SelectedFile, SelectedDataset, dsCoords(SelectedFile, SelectedDataset)),
                           numOfCoords=len(dsCoords(SelectedFile, SelectedDataset)))




def hv_worker():
    asyncio.set_event_loop(asyncio.new_event_loop())
    bokeh_tornado = BokehTornado({'/hvapp': hvapp}, extra_websocket_origins=["127.0.0.1:8000"])
    bokeh_http = HTTPServer(bokeh_tornado)
    bokeh_http.add_sockets(sockets)

    server = BaseServer(IOLoop.current(), bokeh_tornado, bokeh_http)
    server.start()
    server.io_loop.start()


@app.route('/propos')
def propos():
    return render_template("propos.html")


t = Thread(target=hv_worker)
t.daemon = True
t.start()

if __name__ == '__main__':
    print('This script is intended to be run with gunicorn. e.g.')
    print()
    print('    gunicorn -w 4 flaskAppMultiThread2:app')
    print()
    print('will start the app on four processes')
    import sys

    sys.exit()
