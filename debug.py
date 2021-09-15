# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:38:41 2021

@author: sandipto.sanyal
"""

from google.cloud import storage
from pathlib import Path


bucket_name = 'foundation_project2'
prefix = 'bin_files/model/v20210914150518'
dl_dir = './bin_files'

storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)
blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
for blob in blobs:
    if blob.name.endswith("/"):
        continue
    file_split = blob.name.split("/")
    directory = "/".join(file_split[0:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(blob.name) 