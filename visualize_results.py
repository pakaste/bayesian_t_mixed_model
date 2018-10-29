import os
import sys

from os.path import split
from os.path import join
from os.path import abspath
from os.path import dirname
from os import pardir

root_dir = split(os.getcwd())[0]

if root_dir not in sys.path:
    sys.path.append(root_dir)

current_dir = abspath(dirname(__file__))
parent_dir = abspath(join(current_dir, os.pardir))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

from utils.general import get_paths
from utils.general import get_filename
from settings import CONFIGS as cf



def visualize_results():

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True,
        help="folder of gibbs parameters")
    ap.add_argument("-c", "--chain", required=False,
        help="(int), which chain to visualize. Default=0")
    args = vars(ap.parse_args())

    print('Getting results from folder {}'.format(args["folder"]))
    path = os.path.join(cf.CURRENT_DIR, args["folder"])
    print(path)

    #if args["chain"]:
    #    chain = 'chain_' + str(args["chain"])
    #else:
    #    chain = 'chain_0'

    chain_ids = [0, 1, 2, 3]
            # Read in files and plot the results
    data = []
    initial_values = [1.4966943391686316,
                        0.4919518052559649,
                        0.8615640710975296,
                        0.13052001985399436]

    for idx, chain_id in enumerate(chain_ids):
        chain = 'chain_' + str(chain_id)

        for file in get_paths(path, typelist=['csv'], verbose=False):
            print('file', file)
            cpg_name = get_filename(file).split('_')[0]

            if chain in file:
                print(file)
                if '0' in file:
                    val = round(initial_values[idx], 2)
                elif '1' in file:
                    val = round(initial_values[idx], 2)
                elif '2' in file:
                    val = round(initial_values[idx], 2)
                else:
                    val = round(initial_values[idx], 2)

                df = pd.read_csv(file, header=None)

                if 'coefficients' in file:
                    i = 1   # age

                    # Subset data
                    estimate = df.loc[cf.BURN_IN:, i]
                    subset = estimate[::20]  # 0.00320705699501159
                    print('\nEstimated param (median) for subset = ', round(np.nanmedian(subset), 5))
                    print('\nEstimated param (mean) for subset = ', round(np.nanmean(subset), 5))
                    print('\nEstimated param (std) for subset = ', round(np.nanstd(subset), 5))

                    # Create iplotly
                    trace1 = go.Scatter(
                        x = np.array(range(len(subset)))*20,
                        y = subset,
                        mode = 'lines+markers',
                        name = 'aloitusarvo: ' + str(val)
                    )

                    data.append(trace1)

    # Edit the layout
    layout = dict(title = 'Iän kulmakertoimet ' + cpg_name + ':lle eri aloitusarvoilla',
                  xaxis = dict(title = 'iteraatio'),
                  yaxis = dict(title = 'kulmakerroin'),
                  )

    fig = dict(data=data, layout=layout)
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig,show_link = False, image_width=1200, image_height=800)



def _main():
    visualize_results()


if __name__ == '__main__':
    _main()
