"""
download.py
This script provides automatic download methods to get the training materials associated with the demo.
Methods will be called within training and predictions scripts if repositories are missing.
Running this script directly will download the full set of data.
"""

from DeepPhysX_Core.Utils.data_downloader import DataDownloader

DOI = ''
session_name = 'liver_dpx'


class LiverDownloader(DataDownloader):

    def __init__(self):
        DataDownloader.__init__(self, DOI, session_name)

        self.categories = {'models': [],
                           'session': [],
                           'network': [],
                           'stats': [],
                           'dataset_info': [],
                           'dataset_valid': [],
                           'dataset_train': []}


if __name__ == '__main__':

    LiverDownloader().get_session('all')
