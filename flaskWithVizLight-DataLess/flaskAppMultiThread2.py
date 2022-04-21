#flaskAppMultiThread.py

try:
    import asyncio
except ImportError:
    raise RuntimeError("This example requries Python3 / asyncio")

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

from bokeh.server.server import BaseServer
from bokeh.server.tornado import BokehTornado
from bokeh.server.util import bind_sockets

from threading import Thread

from bokeh.client import pull_session
from bokeh.embed import server_session
from bokeh.embed import server_document
from bokeh.server.util import bind_sockets

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

from flask import Flask, render_template
import panel as pn
from flask import send_from_directory

from flask import Flask, render_template, request
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

import holoviews as hv
import panel as pn
import numpy as np

import xarray as xr

test = [1,2,3,4]
hv.extension('bokeh')

#from flask.ext.widgets import Widgets

app = Flask(__name__)

#widgets = Widgets(app)
import os
path = "data/"

datasets = {}
files = os.listdir(path)
for f in files:
    datasets[f] = xr.open_dataset('data/'+f)

fileNames = list(datasets.keys())

def fileKeys(file):
    return list(file.keys())

def dsCoords(dataset):
    return list(dataset.coords)

def coordsValues(file, dataset, coords):
    numOfVal = []
    for coord in coords:
        numOfVal.append(len(datasets[file][dataset].isel()[coord]))
    return numOfVal
def Images(file, dataset, coords, values):
    coords.remove('lat')
    coords.remove('lon')
    isels = {}
    for i in range(len(coords)):
        isels[str(coords[i])] = values[i]
    xarrToPlot = datasets[file][dataset].isel(isels)
    image = hv.Image(xarrToPlot, kdims=["lon", "lat"])
    return image.opts(colorbar=True, height=500, width=800)



def viz(doc):
    #ranges = dict(frequency=(1, 5), phase=(-np.pi, np.pi), amplitude=(-2, 2), y=(-2, 2))
    #dmap = hv.DynamicMap(sine, kdims=['frequency', 'phase', 'amplitude']).redim.range(**ranges)
    image = Images( ... )
    model = pn.Column(image).get_root()
    doc.add_root(model)

sockets, port = bind_sockets("localhost", 0)

hvapp = Application(FunctionHandler(viz))
# locally creates a page
@app.route('/', methods=['GET','POST'])
def hv_page():
    #just default values
    SelectedFile = fileNames[0]
    SelectedDataset = fileKeys(SelectedFile)[0]
    SelectedCoords = dsCoords(SelectedDataset)[0]
    SelectedValues = coordsValues(SelectedFile, SelectedDataset, SelectedCoords)[0]
    if request.method == 'POST':
        freq = request.form['frequency']
    script = server_document('http://localhost:%d/hvapp' % port)
    return render_template("index.html", script=script, template="Flask",
                           files=fileNames, savedFileOpt= SelectedFile,
                           dataset=fileKeys(SelectedFile), savedDsOpt=SelectedDataset,
                           coords=dsCoords(SelectedDataset), savedCoordOpt=SelectedCoords,
                           values=coordsValues(SelectedFile, SelectedDataset, SelectedCoords), )

'''
def dataCollector():
    if request.method == 'POST':
        return float(request.form['frequency']), float(request.form['phase']), float(request.form['amplitude'])
'''


def hv_worker():
    asyncio.set_event_loop(asyncio.new_event_loop())
    # probablement pas marcher
    bokeh_tornado = BokehTornado({'/hvapp': hvapp}, extra_websocket_origins=["127.0.0.1:8000"])
    bokeh_http = HTTPServer(bokeh_tornado)
    bokeh_http.add_sockets(sockets)

    server = BaseServer(IOLoop.current(), bokeh_tornado, bokeh_http)  # peut etre quand meme utiliser bokeh
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