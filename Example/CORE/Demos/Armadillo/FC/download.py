"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

import os

from DeepPhysX_Core.Utils.data_downloader import DataDownloader


class ArmadilloDownloader(DataDownloader):

    def __init__(self, DOI):
        DataDownloader.__init__(self, DOI)

        self.sessions = {'data': 'armadillo_data_dpx',
                         'train': 'armadillo_training_dpx',
                         'model': 'models'}
        self.tree = {'armadillo_data_dpx': [[],
                                            {'dataset': [187, 191, 188, 186, 190]}],
                     'armadillo_training_dpx': [[189],
                                                {'dataset': [],
                                                 'network': [193],
                                                 'stats': [192]}],
                     'models': [[111, 112, 210], {}]}
        self.nb_files = {'data': 5, 'train': 3, 'model': 3}


def download_all():
    downloader = ArmadilloDownloader('doi:10.5072/FK2/B1NUY0')
    downloader.get_session('data')
    downloader.get_session('train')
    downloader.root = os.path.abspath(os.path.join(downloader.root, os.path.pardir, 'Environment'))
    downloader.get_session('model')


if __name__ == '__main__':
    download_all()
