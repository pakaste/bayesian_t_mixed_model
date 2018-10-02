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

from settings import CONFIGS as cf


def visualize_results():

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True,
        help="folder of gibbs parameters")
    ap.add_argument("-c", "--chain", required=False,
        help="(int), which chain to visualize. Default=0")
    args = vars(ap.parse_args())

    print('Getting results from folder {}'.format(args["folder"]))

    files = []
    path = os.path.join(cf.CURRENT_DIR, args["folder"])

    if args["chain"]:
        chain = 'chain_' + str(args["chain"])
    else:
        chain = 'chain_0'

    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and chain in i:
            file_path = os.path.join(path, i)
            files.append(file_path)

    # Read in files and plot the results
    for file in files:
        print(file)
        df = pd.read_csv(file, header=None)

        if 'coefficients' in file:
            for i in range(0, 10):
                # Subset data
                estimate = df.loc[cf.BURN_IN:, i]
                subset = estimate[::20]  # 0.00320705699501159
                print('\nEstimated param for subset = ', round(np.nanmedian(subset), 5))

                # Visualize the subset
                plt.plot(ubset)
                plt.show()



def _main():
    visualize_results()


if __name__ == '__main__':
    _main()
