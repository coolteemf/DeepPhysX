"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX_Core.Utils.data_downloader import DataDownloader


class ArmadilloDownloader(DataDownloader):

    def __init__(self, DOI):
        DataDownloader.__init__(self, DOI)

        self.sessions = {'data': 'armadillo_data_dpx',
                         'train': 'armadillo_training_dpx'}
        self.tree = {'armadillo_data_dpx': [[],
                                            {'dataset': []}],
                     'armadillo_training_dpx': [[],
                                                {'dataset': [],
                                                 'network': [],
                                                 'stats': []}]}
        self.nb_files = {'data': 0, 'train': 0}


def download_dataset():
    downloader = ArmadilloDownloader('')
    downloader.get_session('data')


def download_training():
    downloader = ArmadilloDownloader('')
    downloader.get_session('train')


def download_all():
    downloader = ArmadilloDownloader('')
    downloader.get_session('data')
    downloader.get_session('train')


if __name__ == '__main__':
    download_all()
