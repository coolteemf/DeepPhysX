"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

import os.path

from DeepPhysX_Core.Utils.data_downloader import DataDownloader


class LiverDownloader(DataDownloader):

    def __init__(self, DOI):
        DataDownloader.__init__(self, DOI)

        self.sessions = {'data': 'liver_data_dpx',
                         'train': 'liver_training_dpx',
                         'model': 'models'}
        self.tree = {'liver_data_dpx': [[],
                                        {'dataset': [114, 121, 122, 123, 124]}],
                     'liver_training_dpx': [[119],
                                            {'dataset': [],
                                             'network': [115],
                                             'stats': [116]}],
                     'models': [[117, 118, 120], {}]}
        self.nb_files = {'data': 5, 'train': 3, 'model': 3}


def download_all():
    downloader = LiverDownloader('doi:10.5072/FK2/ZPFUBK')
    downloader.get_session('data')
    downloader.get_session('train')
    downloader.root = os.path.abspath(os.path.join(downloader.root, os.path.pardir, 'Environment'))
    downloader.get_session('model')


if __name__ == '__main__':
    download_all()
