import requests
import json
import sys
sys.path.append('.')
import log
import os
from tqdm import tqdm
import time

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '{:.1f} {}{}'.format(num, unit, suffix)
        num /= 1024.0
    return '{:.1f} {}{}'.format(num, 'Yi', suffix)

def http_download(local_filename,url):
    bytes_downloaded = 0
    try:
        r = requests.get(url, stream=True, timeout=5)
        r.raise_for_status()
        t_start = time.time()
        with tqdm(total = int(r.headers['Content-Length']), smoothing=0.9) as pbar:
            with open(local_filename, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=102400): 
                    pbar.update(len(chunk))
                    bytes_downloaded += len(chunk)
                    speed = int(bytes_downloaded /(time.time() - t_start))
                    status = '  %s (%s/s)'%(sizeof_fmt(bytes_downloaded), sizeof_fmt(speed))
                    pbar.set_description(status)
                    if chunk:
                        fp.write(chunk)
        return True
    except Exception as err:
        print(err)
        return False

def make_kitti_list():
    url_list = json.load(open('kitti/kitti_raw.json','r'))
    kitti_list = []
    for i,url in tqdm(enumerate(url_list), total=len(url_list), smoothing=0.9):
        fn = url.split('/')[-1]
        response = requests.head(url)
        length = int(response.headers['content-length'])
        log.info(fn,length=length)
        kitti_list.append((url,length))

    json.dump(kitti_list,open('kitti/kitti_list.json','w'))

def download(dl_folder = '/media/james/MyPassport/James/dataset/KITTI/raw/'):
    kitti_list = json.load(open('kitti/kitti_list.json','r'))
    for i,(url,length) in enumerate(kitti_list):
        fn = url.split('/')[-1]
        local_filename = os.path.join(dl_folder,fn)
        if os.path.exists(local_filename):
            file_size = os.path.getsize(local_filename)
            if file_size == length:
                log.debug('skip', fn=fn, length = sizeof_fmt(length))
            else:
                log.debug('size mismatch',length, file_size)
                os.remove(local_filename)

        if not os.path.exists(local_filename):
            log.info('Downloading', local_filename=local_filename)
            http_download(local_filename,url)

if __name__ == "__main__":
    # make_kitti_list()
    download()