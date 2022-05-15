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
from bokeh.layouts import column, row

from holoviews.selection import link_selections
from holoviews.operation import gridmatrix
from holoviews.operation.element import histogram
from holoviews import opts

from flask import Flask, render_template, request
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from threading import Thread

import pandas as pd
import holoviews as hv
import panel as pn
import numpy as np
import xarray as xr
import json

hv.extension('bokeh')

app = Flask(__name__)

import os

path = "clovis_data/"

# creating a dict that contain all the netcdf files
datasets = {}
files = os.listdir(path)
for f in files:
    datasets[f] = xr.open_dataset('clovis_data/' + f)

# Get a list of all file names
fileNames = list(datasets.keys())


# List of function to obtain the structure of the files and dataset
def fileKeys(file):
    return np.array(datasets[file].data_vars)




#np.array(clovis_ds.data_vars)

# Innitiating containero.js
SelectedFile = fileNames[0]
SelectedTempMean = '7'
SelectedVersion = 'v3'

optionData = {
    'file': SelectedFile,
    'tempMean': SelectedTempMean,
    'version': SelectedVersion
}

# writing the data in the json
json_string = json.dumps(optionData)
with open('containero.json', 'w') as outfile:
    outfile.write(json_string)


# Function creating the image
def Images(file, tempMean, version):
    # get and calculate the data necessary for the plot
    coefumis = 1.1985

    cci_sss = datasets[file]['cci_sss_'+tempMean+'_'+version].values
    argo_sss = datasets[file]['argo_sss_'+tempMean+'_'+version].values
    colocs_var = datasets[file]['colocs_var_'+tempMean+'_'+version].values
    colocs_err = datasets[file]['colocs_err_'+tempMean+'_'+version].values

    diffSSS = (cci_sss - argo_sss) / np.sqrt((coefumis * colocs_var) ** 2 + colocs_err ** 2)

    lon = datasets[file]['lon_'+tempMean+'_'+version].values
    lat = datasets[file]['lat_'+tempMean+'_'+version].values

    dikt = {
        'lon': lon,
        'lat': lat,
        'diffSSS': diffSSS
    }

    dfDiff = pd.DataFrame(dikt)

    colName = ['lon', 'lat']
    # get lon, lat data into an array of tuples
    value_to_grid = dfDiff[colName].to_numpy()

    #arrays setting the pixels resolution (crucial)
    xRes = np.arange(-180, 180, 0.5)
    yRes = np.arange(-69, 81, 0.25)

    # all the lat and lon will be located to a bin
    bins = [xRes, yRes]
    digitized = []
    for i in range(len(bins)):
        digitized.append(np.digitize(value_to_grid[:, i], bins[i], right=False))
    # the lat and lon are located starting from 1 to n+1, need to rearrange that into proper indexes
    digitized = np.array(digitized) - 1

    # creating the bins for the 2D histogram
    xedges = np.arange(-180, 180.5, 0.5)
    yedges = np.arange(-69, 81.25, 0.25)

    # sumnumpy is where all calculus will be done
    sumnumpy = np.zeros((720, 600))
    # transform all nan to 0 because any calcul that involve a nan create a nan (exemple 5+ nan = nan)
    # this is sad but necessary for the calculs (the nans will be set back later)
    no_nan_diffSSS = np.nan_to_num(dfDiff['diffSSS'], nan=0)

    # creation of the 2D frequency histogram that will help meaning the final values
    H, xedges, yedges = np.histogram2d(dfDiff['lon'], dfDiff['lat'], bins=(xedges, yedges))
    HT = H.T
    divider = np.where(HT == 0, 1, HT)

    # fill sumnumpy with all the diffSSS values
    # all the diffSSS values located in a bin will be summed into the proper bin
    for i in range(len(digitized[0])):
        sumnumpy[np.where(xRes == xRes[digitized[0][i]])[0][0]][np.where(yRes == yRes[digitized[1][i]])[0][0]] += no_nan_diffSSS[i]

    # Matrix division to mean the diffSSS values (if a bin contain the value of 2 diff SSS data, the value of this bin will be divided by 2
    # and that for all the bins
    dividedNumpy = np.divide(sumnumpy, divider.T)

    # Setting back the nans (like i promised)
    dividedNumpy[dividedNumpy == 0] = np.nan

    # Creating the xarray dataset with the gridded data :) youhou
    griddedNc = xr.Dataset(
        data_vars=dict(
            DiffSSS=(["lon", 'lat'], dividedNumpy)
        ),
        coords=dict(
            lon=(["lon"], (xRes + 0.25)),  # the + is here in order to display the pixels properly with hv.image
            lat=(["lat"], (yRes + 0.125)),
            # since the pixel start from the middle of the coord and not the bottom left, we need to correct that
        )
    )

    # transform the netcdf into hvDataset, crucial for interactivity
    griddedHV = hv.Dataset(griddedNc, ['lon', 'lat'])

    #Creating the linked plots
    diffSSSImage = hv.Image(griddedHV, kdims=["lon", "lat"])
    diffSSSImage.opts(colorbar=True, height=500, width=800, tools=['hover', 'box_select'], cmap='RdBu_r')
    diffSSSHist = histogram(griddedHV, normed=False).opts(color="blue", height=500, width=500)

    mpg_ls = link_selections.instance()

    viz = mpg_ls(diffSSSImage + diffSSSHist)

    return viz


# Bokeh app function
def viz(doc):
    f = open("containero.json")
    selected = json.load(f)
    image = Images(selected['file'],
                   selected['tempMean'],
                   selected['version'])
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
    SelectedTempMean = '7'
    SelectedVersion = 'v3'

    # update the json after submit
    if request.method == 'POST':
        #update the file selected
        SelectedFile = request.form['file']
        SelectedTempMean = request.form['tempMean']
        SelectedVersion = request.form['version']

        optionData = {
            'file': SelectedFile,
            'tempMean': SelectedTempMean,
            'version': SelectedVersion
        }

        json_string = json.dumps(optionData)
        with open('containero.json', 'w') as outfile:
            outfile.write(json_string)

    #script containing the app
    script = server_document('http://localhost:%d/hvapp' % port)
    return render_template("new_index.html", script=script, template="Flask",
                           files=fileNames, savedFileOpt=SelectedFile,
                           tempMeanOpt=['7', '30'], savedTmOpt=SelectedTempMean,
                           versionOpt=['v1', 'v2', 'v3'], savedVersionOpt=SelectedVersion)





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
    print('    gunicorn -w 4 flaskAppMultiThread4:app')
    print()
    print('will start the app on four processes')
    import sys

    sys.exit()
