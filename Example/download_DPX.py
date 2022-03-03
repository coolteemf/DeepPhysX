from pyDataverse.api import NativeApi, DataAccessApi

base_url = 'https://data.loria.fr'
token = '92329f1d-efd4-4dd0-b0fb-7f608f74b0df'

api = NativeApi(base_url, token)
resp = api.get_info_version(True)
print(resp.json())




data_api = DataAccessApi(base_url)
print(data_api)


DOI = 'doi:10.5072/FK2/V8FNDZ'
dataset = api.get_dataset(DOI)
print(dataset)