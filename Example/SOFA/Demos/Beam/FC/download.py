"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

import os.path

from DeepPhysX_Core.Utils.data_downloader import DataDownloader


class BeamDownloader(DataDownloader):

    def __init__(self, DOI):
        DataDownloader.__init__(self, DOI)

        self.sessions = {'data': 'beam_data_dpx',
                         'train': 'beam_training_dpx'}
        self.tree = {'beam_data_dpx': [[],
                                       {'dataset': [218, 217, 220, 214, 213]}],
                     'beam_training_dpx': [[215],
                                           {'dataset': [],
                                            'network': [219],
                                            'stats': [216]}]}
        self.nb_files = {'data': 5, 'train': 3}


def download_all():
    downloader = BeamDownloader('doi:10.5072/FK2/8JZ8HO')
    downloader.get_session('data')
    downloader.get_session('train')


if __name__ == '__main__':
    download_all()
