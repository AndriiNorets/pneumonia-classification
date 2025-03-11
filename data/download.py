import kagglehub
import os
import zipfile


def download_dataset():
    dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Dataset downloaded to:", dataset_path)

    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall("data/raw")

    print("Dataset extracted to data/raw")


if __name__ == "__main__":
    download_dataset()
