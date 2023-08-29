import os
import zipfile
from zipfile import ZipFile
from tqdm import tqdm
import pandas as pd

def prepareDataFrame(csv_paths: list[str]):
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Provided file: {csv_path} does not exist.")
    
    # mapping path to dataframe with full paths to images
    def mapPath2DataFrame(path):
        dataf = pd.read_csv(path)
        folder_path = os.path.dirname(path)
        dataf['fullpath'] = folder_path + '/' + dataf['filename']

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
            try:
                zip_ref.extract(file, dest)
            except zipfile.error as e:
                pass