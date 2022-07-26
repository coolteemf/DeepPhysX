"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX_Core.Utils.data_downloader import DataDownloader

DOI = 'doi:10.5072/FK2/8JZ8HO'
session_name = 'beam_dpx'


class BeamDownloader(DataDownloader):

    def __init__(self,):
        DataDownloader.__init__(self, DOI, session_name)

        self.categories = {'models': [],
                           'session': [266],
                           'network': [219],
                           'stats': [216],
                           'dataset_info': [262],
                           'dataset_valid': [263, 264],
                           'dataset_train': [261, 265]}


if __name__ == '__main__':

    BeamDownloader().get_session('all')
