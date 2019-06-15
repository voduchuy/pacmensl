# Script for downloading and installing Parmetis

import wget
import subprocess
from pathlib import Path
import tarfile as tar


def download(path_to):
    dest_dir = Path(path_to)
    print('downloading metis... ')
    url = 'http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz'
    wget.download(url, str(dest_dir))
    f = tar.open(str(dest_dir/Path('metis-5.1.0.tar.gz')))
    f.extractall(dest_dir)
    print('\n downloading parmetis... ')
    url = 'http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz'
    wget.download(url, str(dest_dir))
    f = tar.open(str(dest_dir/Path('parmetis-4.0.3.tar.gz')))
    f.extractall(dest_dir)



def install(src_path, build_path, install_path):
    src_dir = Path(src_path)
    install_dir = Path(install_path)
    src_dir = src_dir.expanduser()
    install_dir = install_dir.expanduser()
    print('configure metis...')
    subprocess.call(["make", "config", "prefix= "+str(install_dir)], cwd=src_dir/Path('metis-5.1.0'))
    print('build metis...')
    subprocess.call(['make', 'install'], cwd=src_dir/Path('metis-5.1.0'))
    print('configure parmetis...')
    subprocess.call(['make', 'config', 'shared=1', 'prefix='+str(install_dir)], cwd=src_path/Path('parmetis-4.0.3'))
    print('build parmetis...')
    subprocess.call(['make', 'install'], cwd=src_path/Path('parmetis-4.0.3'))


if __name__ == "__main__":
    download_dir = '/Users/huyvo/Codes/software/src/'
    install_dir = '/Users/huyvo/Codes/software/install/'
    download(download_dir)
    install(download_dir, '', install_dir)