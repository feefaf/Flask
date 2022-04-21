#flaskAppMultiThread.py

try:
    import asyncio
except ImportError:
    raise RuntimeError("This example requries Python3 / asyncio")


from threading import Thread

from bokeh.client import pull_session
from bokeh.embed import server_session
from bokeh.server.util import bind_sockets

from flask import Flask, render_template
import panel as pn
from flask import send_from_directory

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature
from bokeh.server.server import BaseServer
from bokeh.server.tornado import BokehTornado
from bokeh.server.util import bind_sockets
from bokeh.themes import Theme

from flask import Flask, render_template
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

import holoviews as hv
import panel as pn
import numpy as np

hv.extension('bokeh')

app = Flask(__name__)

def plot(frequency, phase, amplitude):
    xs = np.linspace(0, np.pi*4)
    return hv.Curve((xs, np.sin(frequency*xs+phase)*amplitude)).options(width=800)

def hvapp(doc):
    ranges = dict(frequency=(1, 5), phase=(-np.pi, np.pi), amplitude=(-2, 2), y=(-2, 2))
    dmap = hv.DynamicMap(plot, kdims=['frequency', 'phase', 'amplitude']).redim.range(**ranges)
    print(dmap.data)
    #sinus_data = list(dmap.data)
    #bok_plot = plot(sinus_data[0],sinus_data[1],sinus_data[2])
    doc.add_root(column(bok_plot))
    doc.theme = Theme(filename="theme.yaml")


hvapp = Application(FunctionHandler(hvapp))

sockets, port = bind_sockets("localhost", 0)

# locally creates a page
#si bug, enlever le methods
@app.route('/', methods=['GET'])
def hv_page():
    script = server_document('http://localhost:%d/hvapp' % port)
    return render_template("embed.html", script=script, template="Flask")


def hv_worker():
    asyncio.set_event_loop(asyncio.new_event_loop())

    # create the server
    bokeh_tornado = BokehTornado({'/hvapp': hvapp}, extra_websocket_origins=["127.0.0.1:8000"])
    bokeh_http = HTTPServer(bokeh_tornado)
    bokeh_http.add_sockets(sockets)

    server = BaseServer(IOLoop.current(), bokeh_tornado, bokeh_http)  # peut etre quand meme utiliser bokeh
    server.start()
    server.io_loop.start()



t = Thread(target=hv_worker)
t.daemon = True
t.start()


if __name__ == '__main__':
    print('This script is intended to be run with gunicorn. e.g.')
    print()
    print('    gunicorn -w 4 flaskAppMultiThread_bokeh:app')
    print()
    print('will start the app on four processes')
    import sys
    sys.exit()