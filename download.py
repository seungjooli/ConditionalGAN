import os
import requests
import tarfile
import urllib.request
import zipfile
from tqdm import tqdm


def maybe_download_from_url(url, download_dir):
    """
    Download the data from url, unless it's already here.

    Args:
        download_dir: string, path to download directory
        url: url to download from

    Returns:
        Path to the downloaded file
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(download_dir, filename)

    os.makedirs(download_dir, exist_ok=True)

    if not os.path.isfile(filepath):
        print('Downloading: "{}"'.format(filepath))
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print('Download complete ({} bytes)'.format(size))
    else:
        print('File already exists: "{}"'.format(filepath))

    return filepath


def maybe_download_from_google_drive(id, filepath):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, filepath, chunk_size=32 * 1024):
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                              unit='B', unit_scale=True, desc=filepath):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    if not os.path.isfile(filepath):
        print('Downloading: "{}"'.format(filepath))
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()

        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, filepath)
        size = os.path.getsize(filepath)
        print('Download complete ({} bytes)'.format(size))
    else:
        print('File already exists: "{}"'.format(filepath))

    return filepath


def maybe_extract(compressed_filepath, train_dir, test_dir):
    def is_image(filepath):
        extensions = ('.jpg', '.jpeg', '.png', '.gif')
        return any(filepath.endswith(ext) for ext in extensions)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print('Extracting: "{}"'.format(compressed_filepath))

    if zipfile.is_zipfile(compressed_filepath):
        with zipfile.ZipFile(compressed_filepath) as zf:
            files = [member for member in zf.infolist() if is_image(member.filename)]
            count = len(files)
            train_test_boundary = int(count * 0.99)
            for i in range(count):
                if i < train_test_boundary:
                    extract_dir = train_dir
                else:
                    extract_dir = test_dir

                if not os.path.exists(os.path.join(extract_dir, files[i].filename)):
                    zf.extract(files[i], extract_dir)
    elif tarfile.is_tarfile(compressed_filepath):
        with tarfile.open(compressed_filepath) as tar:
            files = [member for member in tar if is_image(member.name)]
            count = len(files)
            train_test_boundary = int(count * 0.99)
            for i in range(count):
                if i < train_test_boundary:
                    extract_dir = train_dir
                else:
                    extract_dir = test_dir

                if not os.path.exists(os.path.join(extract_dir, files[i].name)):
                    tar.extract(files[i], extract_dir)
    else:
        raise NotImplemented

    print('Extraction complete')
