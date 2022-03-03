from pyDataverse.api import NativeApi, DataAccessApi

base_url = 'https://dataverse.harvard.edu/'

api = NativeApi(base_url)
data_api = DataAccessApi(base_url)

DOI = 'doi:10.7910/DVN/KBHLOD'
dataset = api.get_dataset(DOI)

files_list = dataset.json()['data']['latestVersion']['files']

for file in files_list:
    filename = file['dataFile']['filename']
    file_id = file['dataFile']['id']
    print(f"Filename {filename}, id {file_id}")
    response = data_api.get_datafile(file_id)
    with open(filename, 'wb') as f:
        f.write(response.content)
