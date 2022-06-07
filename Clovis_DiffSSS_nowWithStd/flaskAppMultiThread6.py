# flaskAppMultiThread4.py

try:
    import asyncio
except ImportError:
    raise RuntimeError("This example requries Python3 / asyncio")

import os

from bokeh.server.server import BaseServer
from bokeh.server.tornado import BokehTornado
from bokeh.embed import server_document
from bokeh.server.util import bind_sockets
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.layouts import column, row

from holoviews.selection import link_selections
from holoviews.operation.element import histogram
from holoviews import opts
from holoviews.operation.stats import univariate_kde
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
SelectedResolution = 2

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

    diffSSS_Usat_Umist = (cci_sss - argo_sss) / np.sqrt((coefumis * colocs_var) ** 2 + colocs_err ** 2)
    diffSSS7_Usat_only = (cci_sss - argo_sss) / np.sqrt(colocs_err ** 2)
    diffSSS7_Usat_Umis = (cci_sss - argo_sss) / np.sqrt(colocs_var ** 2 + colocs_err ** 2)

    #will be subtracted from their outlier later, thats why they called "pre_"
    pre_lon = datasets[file]['lon_'+tempMean+'_'+version].values
    pre_lat = datasets[file]['lat_'+tempMean+'_'+version].values

    dikt = {
        'lon': pre_lon,
        'lat': pre_lat,
        'diffSSS': diffSSS_Usat_Umist
    }

    pre_dfDiff = pd.DataFrame(dikt)
    # To delete outliers
    dfDiffUsatUmist = pre_dfDiff[((pre_dfDiff['lon'] > -180) & (pre_dfDiff['lon']< 180)) &
                        ((pre_dfDiff['lat']> -69) & (pre_dfDiff['lat']< 81))]

    diktUsatOnly = {
        'lon': pre_lon,
        'lat': pre_lat,
        'diffSSS': diffSSS7_Usat_only
    }

    pre_UsatOnly = pd.DataFrame(diktUsatOnly)
    # To delete outliers
    dfDiffUsatOnly = pre_UsatOnly[((pre_UsatOnly['lon'] > -180) & (pre_UsatOnly['lon'] < 180)) &
                    ((pre_UsatOnly['lat'] > -69) & (pre_UsatOnly['lat'] < 81))]

    diktUmisUsat = {
        'lon': pre_lon,
        'lat': pre_lat,
        'diffSSS': diffSSS7_Usat_Umis
    }

    pre_UmisUsat = pd.DataFrame(diktUmisUsat)
    # To delete outliers
    dfDiffUsatUmis = pre_UmisUsat[((pre_UmisUsat['lon'] > -180) & (pre_UmisUsat['lon'] < 180)) &
                      ((pre_UmisUsat['lat'] > -69) & (pre_UmisUsat['lat'] < 81))]

    ###Courbe de distribution des differences
    # Distribution normale qui servira d'échelle
    mu, sigma = 0, 1

    measured = np.random.normal(mu, sigma, 1000)
    hist = np.histogram(measured, density=True, bins=50)

    x = np.linspace(-4, 4, 1000)
    norm = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    ###Seconde partie des plots, Images et histograms de visualisation de la répartition des differences SSS

    colName = ['lon', 'lat']
    # get lon, lat data into an array of tuples
    value_to_grid = dfDiffUsatUmist[colName].to_numpy()

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
    sumnumpyUsatUmist = np.zeros((len(xRes), len(yRes)))
    sumnumpyUsatOnly = np.zeros((len(xRes), len(yRes)))
    sumnumpyUsatUmis = np.zeros((len(xRes), len(yRes)))
    # std sumnumpys section
    sumVarUsatUmist = np.zeros((len(xRes), len(yRes)))
    sumVarUsatOnly = np.zeros((len(xRes), len(yRes)))
    sumVarUsatUmis = np.zeros((len(xRes), len(yRes)))

    # transform all nan to 0 because any calcul that involve a nan create a nan (exemple 5+ nan = nan)
    # this is sad but necessary for the calculs (the nans will be set back later)
    no_nan_diffSSS_Usat_Umist = np.nan_to_num(dfDiffUsatUmist['diffSSS'], nan = 0)
    no_nan_diffSSS_Usat_Only = np.nan_to_num(dfDiffUsatOnly['diffSSS'], nan = 0)
    no_nan_diffSSS_Usat_Umis = np.nan_to_num(dfDiffUsatUmis['diffSSS'], nan = 0)

    # creation of the 2D frequency histogram that will help meaning the final values
    H, xedges, yedges = np.histogram2d(dfDiffUsatUmist['lon'], dfDiffUsatUmist['lat'], bins=(xedges, yedges))
    divider = np.where(H == 0, 1, H)
    # fill sumnumpy with all the diffSSS values
    # all the diffSSS values located in a bin will be summed into the proper bin
    for i in range(len(digitized[0])):
        sumnumpyUsatUmist[digitized[0][i]][digitized[1][i]] += no_nan_diffSSS_Usat_Umist[i]
        sumnumpyUsatOnly[digitized[0][i]][digitized[1][i]] += no_nan_diffSSS_Usat_Only[i]
        sumnumpyUsatUmis[digitized[0][i]][digitized[1][i]] += no_nan_diffSSS_Usat_Umis[i]

    # Matrix division to mean the diffSSS values (if a bin contain the value of 2 diff SSS data, the value of this bin will be divided by 2
    # and that for all the bins
    dividedNumpyDsssUsatUmist = np.divide(sumnumpyUsatUmist, divider)
    dividedNumpyDsssUsatOnly = np.divide(sumnumpyUsatOnly, divider)
    dividedNumpyDsssUsatUmis = np.divide(sumnumpyUsatUmis, divider)

    # now calculating (x - xMean)² where xMean is the mean of the pixel
    for i in range(len(digitized[0])):
        sumVarUsatUmist[digitized[0][i]][digitized[1][i]] += (no_nan_diffSSS_Usat_Umist[i] - dividedNumpyDsssUsatUmist[digitized[0][i]][digitized[1][i]]) ** 2
        sumVarUsatOnly[digitized[0][i]][digitized[1][i]] += (no_nan_diffSSS_Usat_Only[i] - dividedNumpyDsssUsatOnly[digitized[0][i]][digitized[1][i]]) ** 2
        sumVarUsatUmis[digitized[0][i]][digitized[1][i]] += (no_nan_diffSSS_Usat_Umis[i] - dividedNumpyDsssUsatUmis[digitized[0][i]][digitized[1][i]]) ** 2

    #matrix division but now to get the variance
    dividedVarUsatUmist = np.divide(sumVarUsatUmist, divider)
    dividedVarUsatOnly = np.divide(sumVarUsatOnly, divider)
    dividedVarUsatUmis = np.divide(sumVarUsatUmis, divider)
    #transforming variance into std
    dividedStdUsatUmist = np.sqrt(dividedVarUsatUmist)
    dividedStdUsatOnly = np.sqrt(dividedVarUsatOnly)
    dividedStdUsatUmis = np.sqrt(dividedVarUsatUmis)

    # Setting back the nans (like i promised)
    dividedNumpyDsssUsatUmist[dividedNumpyDsssUsatUmist == 0] = np.nan
    dividedNumpyDsssUsatOnly[dividedNumpyDsssUsatOnly == 0] = np.nan
    dividedNumpyDsssUsatUmis[dividedNumpyDsssUsatUmis == 0] = np.nan

    dividedStdUsatUmist[dividedStdUsatUmist == 0] = np.nan
    dividedStdUsatOnly[dividedStdUsatOnly == 0] = np.nan
    dividedStdUsatUmis[dividedStdUsatUmis == 0] = np.nan

    # Creating the xarray dataset with the gridded data
    griddedDSSS_UsatUmist = xr.Dataset(
        data_vars=dict(
            DiffSSS=(["lon", 'lat'], dividedNumpyDsssUsatUmist),
            StdSSS=(["lon", 'lat'], dividedStdUsatUmist)
        ),
        coords=dict(
            lon=(["lon"], (xRes + (res / 2))),  # the + is here in order to display the pixels properly with hv.image
            lat=(["lat"], (yRes + (res / 2))),
            # since the pixel start from the middle of the coord and not the bottom left, we need to correct that
        )
    )

    griddedDSSS_UsatOnly = xr.Dataset(
        data_vars=dict(
            DiffSSS=(["lon", 'lat'], dividedNumpyDsssUsatOnly),
            StdSSS=(["lon", 'lat'], dividedStdUsatOnly)
        ),
        coords=dict(
            lon=(["lon"], (xRes + (res / 2))),
            lat=(["lat"], (yRes + (res / 2))),
        )
    )

    griddedDSSS_UsatUmis = xr.Dataset(
        data_vars=dict(
            DiffSSS=(["lon", 'lat'], dividedNumpyDsssUsatUmis),
            StdSSS=(["lon", 'lat'], dividedStdUsatUmis)
        ),
        coords=dict(
            lon=(["lon"], (xRes + (res / 2))),
            lat=(["lat"], (yRes + (res / 2))),
        )
    )

    # transform the netcdf into hvDataset, crucial for interactivity
    griddedHVdSSS_UsatUmist = hv.Dataset(griddedDSSS_UsatUmist['DiffSSS'], ['lon', 'lat'])
    griddedHVdSSS_UsatOnly = hv.Dataset(griddedDSSS_UsatOnly['DiffSSS'], ['lon', 'lat'])
    griddedHVdSSS_Usat_Umis = hv.Dataset(griddedDSSS_UsatUmis['DiffSSS'], ['lon', 'lat'])
    # same for std values
    griddedHVStdSSS_UsatUmist = hv.Dataset(griddedDSSS_UsatUmist['StdSSS'], ['lon', 'lat'])
    griddedHVStdSSS_UsatOnly = hv.Dataset(griddedDSSS_UsatOnly['StdSSS'], ['lon', 'lat'])
    griddedHVStdSSS_Usat_Umis = hv.Dataset(griddedDSSS_UsatUmis['StdSSS'], ['lon', 'lat'])

    #Creating all the linked plots
    #1- DiffSSS plots
    diffSSSUsatUmistImage = hv.Image(griddedHVdSSS_UsatUmist, kdims=["lon", "lat"])
    diffSSSUsatUmistImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r',title='DSSS/√(𝑈𝑆𝐴𝑇²+𝑈𝑚𝑖𝑠𝑡²)', fontsize={'title':24})
    stdSSSUsatUmistImage = hv.Image(griddedHVStdSSS_UsatUmist, kdims=["lon", "lat"])
    stdSSSUsatUmistImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r',
                               title='ΔDSSS/√(𝑈𝑆𝐴𝑇²+𝑈𝑚𝑖𝑠𝑡²)', fontsize={'title': 24})
    #2- Usat plots
    diffSSSUsatOnlyImage = hv.Image(griddedHVdSSS_UsatOnly, kdims=["lon", "lat"])
    diffSSSUsatOnlyImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r', title='DSSS/√(𝑈𝑆𝐴𝑇²)', fontsize={'title':24})
    stdSSSUsatOnlyImage = hv.Image(griddedHVStdSSS_UsatOnly, kdims=["lon", "lat"])
    stdSSSUsatOnlyImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r',
                               title='ΔDSSS/√(𝑈𝑆𝐴𝑇²+𝑈𝑚𝑖𝑠𝑡²)', fontsize={'title': 24})
    #3- Umist plots
    diffSSSUsatUmisImage = hv.Image(griddedHVdSSS_Usat_Umis, kdims=["lon", "lat"])
    diffSSSUsatUmisImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r', title='DSSS/√(𝑈𝑆𝐴𝑇²+𝑈𝑚𝑖𝑠²)', fontsize={'title':24})
    stdSSSUsatUmisImage = hv.Image(griddedHVStdSSS_Usat_Umis, kdims=["lon", "lat"])
    stdSSSUsatUmisImage.opts(colorbar=True, height=500, width=1000, tools=['hover', 'box_select'], cmap='RdBu_r',
                               title='ΔDSSS/√(𝑈𝑆𝐴𝑇²+𝑈𝑚𝑖𝑠𝑡²)', fontsize={'title': 24})
    mpg_ls = link_selections.instance()

    @param.depends(mpg_ls.param.selection_expr)
    def selection_table(_):
        lon_min = np.min(hv.Table((griddedHVdSSS_UsatUmist.select(mpg_ls.selection_expr).dframe()[['lon']]))['lon'])
        lon_max = np.max(hv.Table((griddedHVdSSS_UsatUmist.select(mpg_ls.selection_expr).dframe()[['lon']]))['lon'])
        lat_min = np.min(hv.Table((griddedHVdSSS_UsatUmist.select(mpg_ls.selection_expr).dframe()[['lat']]))['lat'])
        lat_max = np.max(hv.Table((griddedHVdSSS_UsatUmist.select(mpg_ls.selection_expr).dframe()[['lat']]))['lat'])

        # All means and std (used later as label)
        meanUsatUmist = np.mean(dfDiffUsatUmist ['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))])
        stdUsatUmist = np.std(dfDiffUsatUmist ['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))])

        meanUsatOnly = np.mean(dfDiffUsatOnly['diffSSS'][((dfDiffUsatOnly['lon'] > lon_min) & (dfDiffUsatOnly['lon'] < lon_max)) & (
                    (dfDiffUsatOnly['lat'] > lat_min) & (dfDiffUsatOnly['lat'] < lat_max))])
        stdUsatOnly = np.std(dfDiffUsatOnly['diffSSS'][((dfDiffUsatOnly['lon'] > lon_min) & (dfDiffUsatOnly['lon'] < lon_max)) & (
                    (dfDiffUsatOnly['lat'] > lat_min) & (dfDiffUsatOnly['lat'] < lat_max))])

        meanUsatUmis = np.mean(dfDiffUsatUmist['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))])
        stdUsatUmis = np.std(dfDiffUsatUmist['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))])


        # all the distributions
        distribDsss = dfDiffUsatUmist['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))]
        distribUsatOnly = dfDiffUsatOnly['diffSSS'][((dfDiffUsatOnly['lon'] > lon_min) & (dfDiffUsatOnly['lon'] < lon_max)) & (
                    (dfDiffUsatOnly['lat'] > lat_min) & (dfDiffUsatOnly['lat'] < lat_max))]
        distribUmisAndUsat = dfDiffUsatUmis['diffSSS'][((dfDiffUsatUmis['lon'] > lon_min) & (dfDiffUsatUmis['lon'] < lon_max)) & (
                    (dfDiffUsatUmis['lat'] > lat_min) & (dfDiffUsatUmis['lat'] < lat_max))]

        #Creation of distribution curves
        dSSSplot = hv.Distribution(distribDsss, label='DSSS/√(𝑈_𝑆𝐴𝑇²+𝑈_𝑚𝑖𝑠𝑡²)')
        UsatOnlyplot = hv.Distribution(distribUsatOnly, label='DSSS/√(𝑈_𝑆𝐴𝑇²)')
        UmisAndUsatplot = hv.Distribution(distribUmisAndUsat, label='DSSS/√(𝑈_𝑆𝐴𝑇²+𝑈_𝑚𝑖𝑠²)')

        kdeUsUmt = univariate_kde(dSSSplot, bin_range=(-4, 4), bw_method='silverman', n_samples=200, filled=False).opts(
            height=500, width=500, alpha=1, line_color="#0066ff")
        kdeUs = univariate_kde(UsatOnlyplot, bin_range=(-4, 4), bw_method='silverman', n_samples=200,
                               filled=False).opts(height=500, width=500, alpha=1, line_color="red")
        kdeUsUm = univariate_kde(UmisAndUsatplot, bin_range=(-4, 4), bw_method='silverman', n_samples=200,
                                 filled=False).opts(height=500, width=500, alpha=1, line_color="#ffcc00")

        #the gaussian curve that serve as a comparaison
        normalplot = hv.Curve((x, norm), label='gaussian(μ=0,σ=1)').opts(height=500, width=500, color="black")

        #setting the title that contains the mean and the std of each curve
        kdeUsUmt.opts(title='μ=' + str(meanUsatUmist) + '\nσ=' + str(stdUsatUmist))
        kdeUs.opts(title='μ=' + str(meanUsatOnly) + '\nσ=' + str(stdUsatOnly))
        kdeUsUm.opts(title='μ=' + str(meanUsatUmis) + '\nσ=' + str(stdUsatUmis))

        # kdeCurves contient la superposition de toutes les courbes
        kdeCurves = (normalplot * kdeUsUm * kdeUsUmt * kdeUs)
        kdeCurves.opts(title='ΔSSS Global'+tempMean, legend_position='top_right', height=600, width=1200)

        return pn.Column(kdeUsUmt, kdeUs, kdeUsUm)

    @param.depends(mpg_ls.param.selection_expr)
    def selection_table2(_):
        lon_min = np.min(hv.Table((griddedHVdSSS_UsatUmist.select(mpg_ls.selection_expr).dframe()[['lon']]))['lon'])
        lon_max = np.max(hv.Table((griddedHVdSSS_UsatUmist.select(mpg_ls.selection_expr).dframe()[['lon']]))['lon'])
        lat_min = np.min(hv.Table((griddedHVdSSS_UsatUmist.select(mpg_ls.selection_expr).dframe()[['lat']]))['lat'])
        lat_max = np.max(hv.Table((griddedHVdSSS_UsatUmist.select(mpg_ls.selection_expr).dframe()[['lat']]))['lat'])

        # All means and std (used later as label)
        meanUsatUmist = np.mean(
            dfDiffUsatUmist['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))])
        stdUsatUmist = np.std(
            dfDiffUsatUmist['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))])

        meanUsatOnly = np.mean(
            dfDiffUsatOnly['diffSSS'][((dfDiffUsatOnly['lon'] > lon_min) & (dfDiffUsatOnly['lon'] < lon_max)) & (
                    (dfDiffUsatOnly['lat'] > lat_min) & (dfDiffUsatOnly['lat'] < lat_max))])
        stdUsatOnly = np.std(
            dfDiffUsatOnly['diffSSS'][((dfDiffUsatOnly['lon'] > lon_min) & (dfDiffUsatOnly['lon'] < lon_max)) & (
                    (dfDiffUsatOnly['lat'] > lat_min) & (dfDiffUsatOnly['lat'] < lat_max))])

        meanUsatUmis = np.mean(
            dfDiffUsatUmist['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))])
        stdUsatUmis = np.std(
            dfDiffUsatUmist['diffSSS'][((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))])

        # all the distributions
        distribDsss = dfDiffUsatUmist['diffSSS'][
            ((dfDiffUsatUmist['lon'] > lon_min) & (dfDiffUsatUmist['lon'] < lon_max)) & (
                    (dfDiffUsatUmist['lat'] > lat_min) & (dfDiffUsatUmist['lat'] < lat_max))]
        distribUsatOnly = dfDiffUsatOnly['diffSSS'][
            ((dfDiffUsatOnly['lon'] > lon_min) & (dfDiffUsatOnly['lon'] < lon_max)) & (
                    (dfDiffUsatOnly['lat'] > lat_min) & (dfDiffUsatOnly['lat'] < lat_max))]
        distribUmisAndUsat = dfDiffUsatUmis['diffSSS'][
            ((dfDiffUsatUmis['lon'] > lon_min) & (dfDiffUsatUmis['lon'] < lon_max)) & (
                    (dfDiffUsatUmis['lat'] > lat_min) & (dfDiffUsatUmis['lat'] < lat_max))]

        # Creation of distribution curves
        dSSSplot = hv.Distribution(distribDsss, label='DSSS/√(𝑈_𝑆𝐴𝑇²+𝑈_𝑚𝑖𝑠𝑡²)')
        UsatOnlyplot = hv.Distribution(distribUsatOnly, label='DSSS/√(𝑈_𝑆𝐴𝑇²)')
        UmisAndUsatplot = hv.Distribution(distribUmisAndUsat, label='DSSS/√(𝑈_𝑆𝐴𝑇²+𝑈_𝑚𝑖𝑠²)')

        kdeUsUmt = univariate_kde(dSSSplot, bin_range=(-4, 4), bw_method='silverman', n_samples=200, filled=False).opts(
            height=500, width=500, alpha=1, line_color="#0066ff")
        kdeUs = univariate_kde(UsatOnlyplot, bin_range=(-4, 4), bw_method='silverman', n_samples=200,
                               filled=False).opts(height=500, width=500, alpha=1, line_color="red")
        kdeUsUm = univariate_kde(UmisAndUsatplot, bin_range=(-4, 4), bw_method='silverman', n_samples=200,
                                 filled=False).opts(height=500, width=500, alpha=1, line_color="#ffcc00")

        # the gaussian curve that serve as a comparaison
        normalplot = hv.Curve((x, norm), label='gaussian(μ=0,σ=1)').opts(height=500, width=500, color="black")

        # setting the title that contains the mean and the std of each curve
        kdeUsUmt.opts(title='μ=' + str(meanUsatUmist) + '\nσ=' + str(stdUsatUmist))
        kdeUs.opts(title='μ=' + str(meanUsatOnly) + '\nσ=' + str(stdUsatOnly))
        kdeUsUm.opts(title='μ=' + str(meanUsatUmis) + '\nσ=' + str(stdUsatUmis))

        # kdeCurves contient la superposition de toutes les courbes
        kdeCurves = (normalplot * kdeUsUm * kdeUsUmt * kdeUs)
        kdeCurves.opts(title='ΔSSS Global' + tempMean, legend_position='top_right', height=600, width=1200)

        return kdeCurves

    #Widget that allow us to play with the colobar limits
    #js code to change colobar limit
    jscode = """
            color_mapper.low = cb_obj.value[0];
            color_mapper.high = cb_obj.value[1];
        """
    #All widgets for mean diff SSS
    widgetUsatUmist = pn.widgets.RangeSlider(start=np.min(griddedHVdSSS_UsatUmist['DiffSSS'][~np.isnan(griddedHVdSSS_UsatUmist['DiffSSS'])]),
                                     end=np.max(griddedHVdSSS_UsatUmist['DiffSSS'][~np.isnan(griddedHVdSSS_UsatUmist['DiffSSS'])]))

    linkUsatUmist = widgetUsatUmist.jslink(diffSSSUsatUmistImage, code={'value': jscode})

    widgetUsatOnly = pn.widgets.RangeSlider(start=np.min(griddedHVdSSS_UsatOnly['DiffSSS'][~np.isnan(griddedHVdSSS_UsatOnly['DiffSSS'])]),
                                     end=np.max(griddedHVdSSS_UsatOnly['DiffSSS'][~np.isnan(griddedHVdSSS_UsatOnly['DiffSSS'])]))
    linkUsatOnly = widgetUsatOnly.jslink(diffSSSUsatOnlyImage, code={'value': jscode})

    widgetUsatUmis = pn.widgets.RangeSlider(start=np.min(griddedHVdSSS_Usat_Umis['DiffSSS'][~np.isnan(griddedHVdSSS_Usat_Umis['DiffSSS'])]),
                     end=np.max(griddedHVdSSS_Usat_Umis['DiffSSS'][~np.isnan(griddedHVdSSS_Usat_Umis['DiffSSS'])]))
    linkUsatUmis = widgetUsatUmis.jslink(diffSSSUsatUmisImage, code={'value': jscode})

    #Same but for StdSSS
    widgetStdUsatUmist = pn.widgets.RangeSlider(start=np.min(griddedHVStdSSS_UsatUmist['StdSSS'][~np.isnan(griddedHVStdSSS_UsatUmist['StdSSS'])]),
                                     end=np.max(griddedHVStdSSS_UsatUmist['StdSSS'][~np.isnan(griddedHVStdSSS_UsatUmist['StdSSS'])]))
    linkStdUsatUmist = widgetStdUsatUmist.jslink(stdSSSUsatUmistImage, code={'value': jscode})

    widgetStdUsatOnly = pn.widgets.RangeSlider(start=np.min(griddedHVStdSSS_UsatOnly['StdSSS'][~np.isnan(griddedHVStdSSS_UsatOnly['StdSSS'])]),
                                     end=np.max(griddedHVStdSSS_UsatOnly['StdSSS'][~np.isnan(griddedHVStdSSS_UsatOnly['StdSSS'])]))
    linkStdUsatOnly = widgetStdUsatOnly.jslink(stdSSSUsatOnlyImage, code={'value': jscode})

    widgetStdUsatUmis = pn.widgets.RangeSlider(start=np.min(griddedHVStdSSS_Usat_Umis['StdSSS'][~np.isnan(griddedHVStdSSS_Usat_Umis['StdSSS'])]),
                     end=np.max(griddedHVStdSSS_Usat_Umis['StdSSS'][~np.isnan(griddedHVStdSSS_Usat_Umis['StdSSS'])]))
    linkStdUsatUmis = widgetStdUsatUmis.jslink(stdSSSUsatUmisImage, code={'value': jscode})

    AllDsssImages = mpg_ls(pn.Column(widgetUsatUmist,diffSSSUsatUmistImage) + pn.Column(widgetUsatOnly, diffSSSUsatOnlyImage) + pn.Column(widgetUsatUmis, diffSSSUsatUmisImage))
    AllStdImages = mpg_ls(pn.Column(widgetStdUsatUmist, stdSSSUsatUmistImage) + pn.Column(widgetStdUsatOnly, stdSSSUsatOnlyImage) + pn.Column(widgetStdUsatUmis,stdSSSUsatUmisImage))
    return pn.Column(selection_table2, pn.Row(pn.Tabs(('Mean', AllDsssImages),('Std', AllStdImages)), selection_table))


# Bokeh app function
def viz(doc):
    f = open("containero.json")
    selected = json.load(f)
    vizTotale = Images(
                   selected['file'],
                   selected['tempMean'],
                   selected['version'],
                   selected['resolution'])
    f.close()
    vizTotBokeh = pn.Column(vizTotale).get_root()
    doc.add_root(vizTotBokeh)


sockets, port = bind_sockets("localhost", 0)

hvapp = Application(FunctionHandler(viz))


# locally creates a page
@app.route('/', methods=['GET', 'POST'])
def hv_page():
    # just set default selected values
    SelectedFile = fileNames[0]
    SelectedTempMean = '7'
    SelectedVersion = 'v3'
    SelectedResolution = 2

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
    print('    gunicorn -w 4 flaskAppMultiThread6:app')
    print()
    print('will start the app on four processes')
    import sys

    sys.exit()
