import requests
import tarfile 
import os

def download_zips(url):
    if isinstance(url, dict):
        download_url(url["url"], url["save_path"])
    else:
        raise TypeError
        
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def untar(f_name):
    tar = tarfile.open(f_name)
    tar.extractall()
    tar.close()
    return f_name[:-7] # Works only for tar.gz


if __name__ == "__main__":
    url_dict = {
        "url": "https://zenodo.org/record/3251672/files/cls-acl10-processed.tar.gz?download=1",
        "save_path": "./cls-acl10-processed.tar.gz"
    }
    
    if not os.path.isfile(url_dict["save_path"]):
        download_zips(url_dict)
    if not os.path.isdir(url_dict["save_path"][:-7]):
        url_dict["save_path"] = untar(url_dict["save_path"])