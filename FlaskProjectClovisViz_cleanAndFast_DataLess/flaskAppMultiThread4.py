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
import param

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
SelectedResolution = 0.5

optionData = {
    'file': SelectedFile,
    'tempMean': SelectedTempMean,
    'version': SelectedVersion,
    'resolution': SelectedResolution
}

# writing the data in the json
json_string = json.dumps(optionData)
with open('containero.json', 'w') as outfile:
    outfile.write(json_string)


# Function creating the image
def Images(file, tempMean, version, res):
    # get and calculate the data necessary for the plot
    coefumis = 1.1985

    cci_sss = datasets[file]['cci_sss_'+tempMean+'_'+version].values
    argo_sss = datasets[file]['argo_sss_'+tempMean+'_'+version].values
    colocs_var = datasets[file]['colocs_var_'+tempMean+'_'+version].values
    colocs_err = datasets[file]['colocs_err_'+tempMean+'_'+version].values

    diffSSS = (cci_sss - argo_sss) / np.sqrt((coefumis * colocs_var) ** 2 + colocs_err ** 2)

    #will be subtracted from their outlier later, thats why they called "pre_"
    pre_lon = datasets[file]['lon_'+tempMean+'_'+version].values
    pre_lat = datasets[file]['lat_'+tempMean+'_'+version].values

    dikt = {
        'lon': pre_lon,
        'lat': pre_lat,
        'diffSSS': diffSSS
    }


    pre_dfDiff = pd.DataFrame(dikt)
    # To delete outliers
    dfDiff = pre_dfDiff[((pre_dfDiff['lon']> -180 ) & (pre_dfDiff['lon']< 180)) & ((pre_dfDiff['lat']> -69) & (pre_dfDiff['lat']< 81))]


    Umist = coefumis * colocs_var
    Usat = colocs_err

    diktUsat = {
        'lon': pre_lon,
        'lat': pre_lat,
        'Usat': Usat
    }

    pre_Usat = pd.DataFrame(diktUsat)
    # To delete outliers
    Usat = pre_Usat[((pre_Usat['lon'] > -180) & (pre_Usat['lon'] < 180)) & ((pre_Usat['lat'] > -69) & (pre_Usat['lat'] < 81))]

    diktUmis = {
        'lon': pre_lon,
        'lat': pre_lat,
        'Umist': Umist
    }

    pre_Umist = pd.DataFrame(diktUmis)
    # To delete outliers
    Umist = pre_Umist[((pre_Umist['lon'] > -180) & (pre_Umist['lon'] < 180)) & ((pre_Umist['lat'] > -69) & (pre_Umist['lat'] < 81))]

    colName = ['lon', 'lat']
    # get lon, lat data into an array of tuples
    value_to_grid = dfDiff[colName].to_numpy()

    #getting rid of outliers like i promised
    lon = pre_lon[(pre_lon > -180) & (pre_lon < 180)]
    lat = pre_lat[(pre_lat > -69) & (pre_lat < 81)]

    # get the lon, lat ranges
    lon_floor = np.floor(np.min(lon))
    lon_ceil = np.ceil(np.max(lon))

    lat_floor = np.floor(np.min(lat))
    lat_ceil = np.ceil(np.max(lat))

    #arrays setting the pixels resolution (crucial)
    xRes = np.arange(lon_floor, lon_ceil, res)
    yRes = np.arange(lat_floor, lat_ceil, res)


    # all the lat and lon will be located to a bin
    bins = [xRes, yRes]
    digitized = []
    for i in range(len(bins)):
        digitized.append(np.digitize(value_to_grid[:, i], bins[i], right=False))
    # the lat and lon are located starting from 1 to n+1, need to rearrange that into proper indexes
    digitized = np.array(digitized) - 1

    # creating the bins for the 2D histogram
    xedges = np.arange(lon_floor,lon_ceil+res, res)
    yedges = np.arange(lat_floor,lat_ceil+res, res)

    #Those sumnumpy is where all calculus will be done
    sumnumpyDsss = np.zeros((len(xRes), len(yRes)))
    sumnumpyUsat = np.zeros((len(xRes), len(yRes)))
    sumnumpyUmist = np.zeros((len(xRes), len(yRes)))
    # transform all nan to 0 because any calcul that involve a nan create a nan (exemple 5+ nan = nan)
    # this is sad but necessary for the calculs (the nans will be set back later)
    no_nan_diffSSS = np.nan_to_num(dfDiff['diffSSS'], nan = 0)
    no_nan_Usat = np.nan_to_num(Usat['Usat'], nan = 0)
    no_nan_Umist = np.nan_to_num(Umist['Umist'], nan = 0)
    # creation of the 2D frequency histogram that will help meaning the final values
    H, xedges, yedges = np.histogram2d(dfDiff['lon'], dfDiff['lat'], bins=(xedges, yedges))
    divider = np.where(H == 0, 1, H)

    # fill sumnumpy with all the diffSSS values
    # all the diffSSS values located in a bin will be summed into the proper bin
    for i in range(len(digitized[0])):
        sumnumpyDsss[digitized[0][i]][digitized[1][i]] += no_nan_diffSSS[i]
        sumnumpyUsat[digitized[0][i]][digitized[1][i]] += no_nan_Usat[i]
        sumnumpyUmist[digitized[0][i]][digitized[1][i]] += no_nan_Umist[i]
    # Matrix division to mean the diffSSS values (if a bin contain the value of 2 diff SSS data, the value of this bin will be divided by 2
    # and that for all the bins
    dividedNumpyDsss = np.divide(sumnumpyDsss, divider)
    dividedNumpyUsat = np.divide(sumnumpyUsat, divider)
    dividedNumpyUmist = np.divide(sumnumpyUmist, divider)

    # Setting back the nans (like i promised)
    dividedNumpyDsss[dividedNumpyDsss == 0] = np.nan
    dividedNumpyUsat[dividedNumpyUsat == 0] = np.nan
    dividedNumpyUmist[dividedNumpyUmist == 0] = np.nan

    # Creating the xarray dataset with the gridded data
    griddedDSSS = xr.Dataset(
        data_vars=dict(
            DiffSSS=(["lon", 'lat'], dividedNumpyDsss)
        ),
        coords=dict(
            lon=(["lon"], (xRes + (res / 2))),  # the + is here in order to display the pixels properly with hv.image
            lat=(["lat"], (yRes + (res / 2))),
            # since the pixel start from the middle of the coord and not the bottom left, we need to correct that
        )
    )

    griddedUsat = xr.Dataset(
        data_vars=dict(
            Usat=(["lon", 'lat'], dividedNumpyUsat)
        ),
        coords=dict(
            lon=(["lon"], (xRes + (res / 2))),
            lat=(["lat"], (yRes + (res / 2))),
        )
    )

    griddedUmist = xr.Dataset(
        data_vars=dict(
            Umist=(["lon", 'lat'], dividedNumpyUmist)
        ),
        coords=dict(
            lon=(["lon"], (xRes + (res / 2))),
            lat=(["lat"], (yRes + (res / 2))),
        )
    )

    # transform the netcdf into hvDataset, crucial for interactivity
    griddedHVdSSS = hv.Dataset(griddedDSSS,['lon','lat'])
    griddedHVusat = hv.Dataset(griddedUsat,['lon','lat'])
    griddedHVumist = hv.Dataset(griddedUmist,['lon','lat'])

    #Creating all the linked plots
    #1- DiffSSS plots
    diffSSSImage = hv.Image(griddedHVdSSS, kdims=["lon", "lat"], label=str(np.mean(np.nan_to_num(griddedHVdSSS['DiffSSS'], nan=0))))
    diffSSSImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r')
    diffSSSHist = histogram(griddedHVdSSS, normed=False).opts(color="blue", height=500, width=500)
    #text = hv.Text(5,30000, str(np.mean(np.nan_to_num(griddedHVdSSS['DiffSSS'], nan=0))))
    mpg_ls_sss = link_selections.instance()

    @param.depends(mpg_ls_sss.param.selection_expr)
    def selection_table_diffSSS(_):
        lon_min = np.min(hv.Table((griddedHVdSSS.select(mpg_ls_sss.selection_expr).dframe()[['lon']]))['lon'])
        lon_max = np.max(hv.Table((griddedHVdSSS.select(mpg_ls_sss.selection_expr).dframe()[['lon']]))['lon'])
        lat_min = np.min(hv.Table((griddedHVdSSS.select(mpg_ls_sss.selection_expr).dframe()[['lat']]))['lat'])
        lat_max = np.max(hv.Table((griddedHVdSSS.select(mpg_ls_sss.selection_expr).dframe()[['lat']]))['lat'])

        mean = np.mean(dfDiff['diffSSS'][((dfDiff['lon'] > lon_min) & (dfDiff['lon'] < lon_max)) & (
                    (dfDiff['lat'] > lat_min) & (dfDiff['lat'] < lat_max))])
        std = np.std(dfDiff['diffSSS'][((dfDiff['lon'] > lon_min) & (dfDiff['lon'] < lon_max)) & (
                    (dfDiff['lat'] > lat_min) & (dfDiff['lat'] < lat_max))])
        return hv.Table({'mean': mean, 'std': std}, ['mean', 'std']).opts(width=200, height=200)
    vizDiffSSS = pn.Row(mpg_ls_sss(diffSSSImage + diffSSSHist), selection_table_diffSSS)

    #2- Usat plots
    UsatImage = hv.Image(griddedHVusat, kdims=["lon", "lat"])
    UsatImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r')
    UsatHist = histogram(griddedHVusat, normed=False).opts(color="red", height=500, width=500)

    mpg_ls_usat = link_selections.instance()

    @param.depends(mpg_ls_usat.param.selection_expr)
    def selection_table_Usat(_):
        lon_min = np.min(hv.Table((griddedHVusat.select(mpg_ls_usat.selection_expr).dframe()[['lon']]))['lon'])
        lon_max = np.max(hv.Table((griddedHVusat.select(mpg_ls_usat.selection_expr).dframe()[['lon']]))['lon'])
        lat_min = np.min(hv.Table((griddedHVusat.select(mpg_ls_usat.selection_expr).dframe()[['lat']]))['lat'])
        lat_max = np.max(hv.Table((griddedHVusat.select(mpg_ls_usat.selection_expr).dframe()[['lat']]))['lat'])

        mean = np.mean(Usat['Usat'][((Usat['lon'] > lon_min) & (Usat['lon'] < lon_max)) & (
                (Usat['lat'] > lat_min) & (Usat['lat'] < lat_max))])
        std = np.std(Usat['Usat'][((dfDiff['lon'] > lon_min) & (Usat['lon'] < lon_max)) & (
                (Usat['lat'] > lat_min) & (Usat['lat'] < lat_max))])
        return hv.Table({'mean': mean, 'std': std}, ['mean', 'std']).opts(width=200, height=200)

    vizUsat = pn.Row(mpg_ls_usat(UsatImage + UsatHist), selection_table_Usat)


    #3- Umist plots
    UmistImage = hv.Image(griddedHVumist, kdims=["lon", "lat"])
    UmistImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r')
    UmistHist = histogram(griddedHVumist, normed=False).opts(color="yellow", height=500, width=500)

    mpg_ls_umist = link_selections.instance()

    @param.depends(mpg_ls_umist.param.selection_expr)
    def selection_table_Umist(_):
        lon_min = np.min(hv.Table((griddedHVumist.select(mpg_ls_umist.selection_expr).dframe()[['lon']]))['lon'])
        lon_max = np.max(hv.Table((griddedHVumist.select(mpg_ls_umist.selection_expr).dframe()[['lon']]))['lon'])
        lat_min = np.min(hv.Table((griddedHVumist.select(mpg_ls_umist.selection_expr).dframe()[['lat']]))['lat'])
        lat_max = np.max(hv.Table((griddedHVumist.select(mpg_ls_umist.selection_expr).dframe()[['lat']]))['lat'])

        mean = np.mean(Umist['Umist'][((Umist['lon'] > lon_min) & (Umist['lon'] < lon_max)) & (
                (Umist['lat'] > lat_min) & (Umist['lat'] < lat_max))])
        std = np.std(Umist['Umist'][((dfDiff['lon'] > lon_min) & (Umist['lon'] < lon_max)) & (
                (Umist['lat'] > lat_min) & (Umist['lat'] < lat_max))])
        return hv.Table({'mean': mean, 'std': std}, ['mean', 'std']).opts(width=200, height=200)

    vizUmist = pn.Row(mpg_ls_umist(UmistImage + UmistHist), selection_table_Umist)

    return vizDiffSSS, vizUsat, vizUmist, griddedHVdSSS


# Bokeh app function
def viz(doc):
    f = open("containero.json")
    selected = json.load(f)
    vizDiffSSS, vizUsat, vizUmist, diffDF = Images(
                   selected['file'],
                   selected['tempMean'],
                   selected['version'],
                   selected['resolution'])
    f.close()
    model = pn.Column(vizDiffSSS, vizUsat, vizUmist).get_root()
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
    SelectedResolution = 0.5

    # update the json after submit
    if request.method == 'POST':
        #update the file selected
        SelectedFile = request.form['file']
        SelectedTempMean = request.form['tempMean']
        SelectedVersion = request.form['version']
        SelectedResolution = float(request.form['resSlider'])

        optionData = {
            'file': SelectedFile,
            'tempMean': SelectedTempMean,
            'version': SelectedVersion,
            'resolution': SelectedResolution
        }

        json_string = json.dumps(optionData)
        with open('containero.json', 'w') as outfile:
            outfile.write(json_string)

    #script containing the app
    script = server_document('http://localhost:%d/hvapp' % port)
    return render_template("new_index.html", script=script, template="Flask",
                           files=fileNames, savedFileOpt=SelectedFile,
                           tempMeanOpt=['7', '30'], savedTmOpt=SelectedTempMean,
                           versionOpt=['v2', 'v3'], savedVersionOpt=SelectedVersion,
                           sliderValue=SelectedResolution)





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
