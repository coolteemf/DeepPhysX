import os
from pyDataverse.api import NativeApi, DataAccessApi


class DataDownloader:

    def __init__(self, DOI):

        # Connect to Dataverse API
        print("Connecting to DeepPhysX online storage...")
        base_url = 'https://data-qualif.loria.fr'
        self.api = NativeApi(base_url)
        self.data_api = DataAccessApi(base_url)

        # Get files data for desired Dataset
        self.files = self.get_filenames(DOI)

        # Sessions repositories and content
        self.root = os.path.abspath('sessions')
        self.sessions = {'data': '',
                         'train': ''}
        self.tree = {'session_name': [[],
                                      {'subdirectory': []}]}
        self.nb_files = {'all': 0, 'data': 0, 'train': 0}
        self.nb_loaded = 0

    def get_filenames(self, DOI):

        # Get file data for each file in Dataset
        filenames = {}
        for file in self.api.get_dataset(DOI).json()['data']['latestVersion']['files']:
            filenames[file['dataFile']['id']] = file['dataFile']['filename']
        return filenames

    def check_tree(self, session_name):

        # Check root repository
        if not os.path.exists(self.root):
            os.mkdir(self.root)

        # Check desired session repository
        session_dir = os.path.join(self.root, session_name)
        if not os.path.exists(session_dir):
            os.mkdir(session_dir)
            # Create each sub repository
            for repo in self.tree[session_name][1].keys():
                # If no training dataset, link to existing dataset
                if repo == 'dataset' and len(self.tree[session_name][1][repo]) == 0 and 'data' in self.sessions.keys():
                    os.symlink(os.path.join(self.root, self.sessions['data'], repo),
                               os.path.join(session_dir, repo))
                # In all other case create directory
                else:
                    os.mkdir(os.path.join(session_dir, repo))

    def download_files(self, file_list, repository, nb_files):

        # Download each file of the list
        for file_id in file_list:
            self.nb_loaded += 1
            print(f"\tDownloading file {self.nb_loaded}/{nb_files} in "
                  f"{os.path.join(repository[len(os.getcwd()):], self.files[file_id])}")
            file = self.data_api.get_datafile(file_id)
            with open(os.path.join(repository, self.files[file_id]), 'wb') as f:
                f.write(file.content)

    def get_session(self, session_type):

        # Check session tree
        session_name = self.sessions[session_type]
        self.check_tree(session_name)

        # Download files in session repository
        self.download_files(file_list=self.tree[session_name][0],
                            repository=os.path.join(self.root, session_name),
                            nb_files=self.nb_files[session_type])

        # Download files in sub repositories
        for repo, files in self.tree[session_name][1].items():
            self.download_files(file_list=files,
                                repository=os.path.join(self.root, session_name, repo),
                                nb_files=self.nb_files[session_type])

    def show_content(self):

        # Print the content of the desired Dataset
        for file_id in sorted(list(self.files.keys())):
            print(f"\t{file_id}: {self.files[file_id]}")
