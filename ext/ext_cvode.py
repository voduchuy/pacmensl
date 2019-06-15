# Functions for downloading, building and installing armadillo

import wget
import subprocess
from pathlib import Path
import tarfile as tar


def download(path_to):
    dest_dir = Path(path_to)
    dest_dir = dest_dir.expanduser()
    url = 'https://computation.llnl.gov/projects/sundials/download/sundials-4.1.0.tar.gz'
    wget.download(url, str(dest_dir))
    f = tar.open(dest_dir/Path('sundials-4.1.0.tar.gz'))
    f.extractall(dest_dir)


def install(src_path, build_path, install_path):
    src_dir = Path(src_path) / Path('sundials-4.1.0')
    build_dir = Path(build_path) / Path('sundials')
    install_dir = Path(install_path)
    src_dir = src_dir.expanduser()
    build_dir = build_dir.expanduser()
    install_dir = install_dir.expanduser()
    if not build_dir.exists():
        build_dir.mkdir()

    subprocess.call(['cmake', '-DCMAKE_INSTALL_PREFIX=' + str(install_dir), str(src_dir),
                     '-DPETSC_ENABLE=ON',
                     '-DMPI_ENABLE=ON',
                     str(src_dir)], cwd=build_dir)
    subprocess.call(['make'], cwd=build_dir)
    subprocess.call(['make', 'install'], cwd=build_dir)


if __name__ == "__main__":
    download_dir = '/Users/huyvo/Codes/software/src/'
    build_dir = '~/Codes/software/build/'
    install_dir = '/Users/huyvo/Codes/software/install/'
    download(download_dir)
    install(download_dir, build_dir, install_dir)
