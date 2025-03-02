import tarfile
import requests
from pathlib import Path

data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
downloads_path = './downloads'
uncompressed_path = 'Old/data/cifar-10'

def download_data(url, destination):
    """Downloads the requested data, only if it was not downloaded yet."""
    Path(destination).mkdir(exist_ok=True)
    destination = Path(destination) / Path(url).name
    if not destination.is_file():
        response = requests.get(url)
        if response.status_code == 200:
            with open(destination, 'wb') as file:
                file.write(response.content)
    return destination

def uncompress(filename, destination):
    """ Uncompresses a tar.gz file to the specified directory"""
    Path(destination).mkdir(parents=True, exist_ok=True)
    file = tarfile.open(filename)
    file.extractall(destination)
    file.close()

compressed_file = download_data(data_url, downloads_path)
uncompress(compressed_file, uncompressed_path)