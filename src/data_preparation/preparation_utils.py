import os
import zipfile
from zipfile import ZipFile
from tqdm import tqdm
import gdown
import pandas as pd


def prepare_data_frame(csv_paths: list[str], images_path: str):
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Provided file: {csv_path} does not exist.")
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"Provided directory: {images_path} does not exist.")

    # mapping path to dataframe with full paths to images
    def map_path_to_data_frame(csv_path):
        dataf = pd.read_csv(csv_path)
        fullpaths = images_path + "/" + dataf["filename"]
        for path in fullpaths:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"The image: {path}, from dataframe: {csv_path} does not exist."  # noqa: E501
                )
        dataf["fullpath"] = fullpaths

        return dataf

    dataframes = list(map(lambda x: map_path_to_data_frame(x), csv_paths))
    return pd.concat(dataframes)


def zip_to_folder(src: str, dest: str):
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Provided file: {src} does not exist.")
    if not os.path.isdir(dest):
        raise FileNotFoundError(f"Provided directory: {dest} does not exist.")

    filename = os.path.basename(src)
    with ZipFile(src, "r") as zip_ref:
        for file in tqdm(zip_ref.infolist(), desc=f"Extracting {filename}: "):
            fname = os.path.basename(file.filename)
            if fname:
                file.filename = fname
                try:
                    zip_ref.extract(file, dest)
                except zipfile.error:
                    pass


def download_g_drive_file(id: str, dest_file_path: str):
    prefix = "https://drive.google.com/uc?/export=download&id="
    url = prefix + id
    return gdown.download(url=url, output=dest_file_path)


def download_g_drive_folder(url: str, dest_folder_path: str):
    return gdown.download_folder(url=url, output=dest_folder_path)
