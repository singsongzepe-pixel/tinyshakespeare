import requests
import os

from constants import *
from utils import *


def download_tiny_shakespeare():
    log_info(f'start to request dataset file: {TINY_SHAKESPEARE_URL}')
    req = requests.get(TINY_SHAKESPEARE_URL)

    filename = os.path.join(file_path, TINY_SHAKESPEARE_FILENAME)    
    if req.status_code == 200:
        with open(filename, 'w') as pf:
            pf.write(req.text)

        log_info(f'file download successfully: {TINY_SHAKESPEARE_URL}')
    else:
        log_warn(f'file download failed, request status code: {req.status_code}')

def main():
    # check whether file is downloaded
    if not (os.path.exists(file_path) and os.path.isdir(file_path)):
        # dir exists
        os.mkdir(file_path)
        log_info(f'dir at path: {file_path} not exists, now created')
    
    filename = os.path.join(file_path, TINY_SHAKESPEARE_FILENAME)
    if not os.path.exists(filename):
        # file not exist
        # download it
        download_tiny_shakespeare()
    else:
        debug_info(f'file: {filename} already exists, but pay attention to the file is completely downloaded')

if __name__ == '__main__':
    main()

