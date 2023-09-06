import os
import zipfile
from zipfile import ZipFile
from tqdm import tqdm
import gdown
import pandas as pd

def prepareDataFrame(csv_paths: list[str], images_path: str):
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Provided file: {csv_path} does not exist.")
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"Provided directory: {images_path} does not exist.")
    
    # mapping path to dataframe with full paths to images
    def mapPath2DataFrame(csv_path):
        dataf = pd.read_csv(csv_path)
        fullpaths = images_path + '/' + dataf['filename']
        for path in fullpaths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"The image: {path}, from dataframe: {csv_path} does not exist.")
        dataf['fullpath'] = fullpaths

        return dataf

    dataframes = list(map(lambda x: mapPath2DataFrame(x), csv_paths))
    return pd.concat(dataframes)
    
    
    
def zip2folder(src: str, dest: str):
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Provided file: {src} does not exist.")
    if not os.path.isdir(dest):
        raise FileNotFoundError(f"Provided directory: {dest} does not exist.")

    with ZipFile(src, 'r') as zip_ref:
        for file in tqdm(zip_ref.infolist(), desc='Extracting '):
            fname = os.path.basename(file.filename)
            if fname:
                file.filename = fname
                try:
                    zip_ref.extract(file, dest)
                except zipfile.error as e:
                    pass

def downloadGDriveFile(id: str, dest_file_path: str):
    prefix = 'https://drive.google.com/uc?/export=download&id='
    url = prefix + id
    gdown.download(url = url, output=dest_file_path)