import os
from pathlib import Path


def create_folder_ifn_exist(folder):
    folder = Path(folder).absolute()
    if Path(folder).suffix:
        folder = folder.parent
    if not os.path.exists(folder):
        os.makedirs(folder)