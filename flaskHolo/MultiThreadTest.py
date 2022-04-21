import panel as pn, numpy as np
import holoviews as hv, pandas as pd
import asyncio
import time

from threading import Thread  # Asyncio
from functools import partial
from holoviews.streams import Pipe, Buffer

pn.extension()
pn.config.sizing_mode = 'stretch_width'
hv.extension('bokeh')

# Example df for buffer
df = pd.DataFrame({"x": np.array([]),
                   "y": np.array([]),
                   "z": np.array([])}).set_index("x")

# Buffer that updates plot
buffer = Buffer(data=df)

# Some example scatter plot
dmap = hv.DynamicMap(hv.Scatter,
                     streams=[buffer]).opts(bgcolor='black',
                                            color='z',
                                            show_legend=False,
                                            width=1200, height=800, responsive=False
                                            )


@asyncio.coroutine
def update(x, y, z):
    buffer.send(pd.DataFrame({"x": x, "y": y, "z": z}).set_index("x"))


def blocking_task(doc):
    time.sleep(1)
    x = np.random.rand(1)
    y = np.random.rand(1)
    z = np.random.rand(1)
    # update the document from callback
    if doc:
        doc.add_next_tick_callback(partial(update, x=x, y=y, z=z))


def button_click(event):
    thread = Thread(target=partial(blocking_task, doc=pn.state.curdoc))
    thread.start()


btn = pn.widgets.Button(name='Run')
btn.on_click(button_click)

p1 = pn.Column(btn,
               pn.Row(dmap, width=1200, height=800, sizing_mode="fixed")
               )
p1.show('streaming hv')