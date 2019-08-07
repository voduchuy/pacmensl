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

    subprocess.call(['cmake', '-DCMAKE_INSTALL_PREFIX=' + str(install_dir.resolve()), str(src_dir.resolve()),
                     '-DPETSC_ENABLE=ON',
                     '-DMPI_ENABLE=ON',
                     '-DPETSC_INCLUDE_DIR=' + str((install_dir/Path('include')).resolve()),
                     '-DPETSC_LIBRARY_DIR=' + str((install_dir/Path('include')).resolve()),
                     str(src_dir.resolve())], cwd=build_dir)
    subprocess.call(['make'], cwd=build_dir)
    subprocess.call(['make', 'install'], cwd=build_dir)


if __name__ == "__main__":
    download_path = Path('something that does not exist')
    build_path = Path('something that does not exist')
    install_path = Path('something that does not exist')

    while not download_path.exists():
        download_path = input('Enter directory path to extract all downloaded source codes:')
        download_path = Path(download_path).expanduser()
        if not download_path.exists():
            print('Not a valid path, enter again.')

    while not build_path.exists():
        build_path = input('Enter directory path to do compilation (must be different from source code directory):')
        build_path = Path(build_path).expanduser()
        if not build_path.exists():
            print('Not a valid build path, enter again.')

    while not install_path.exists():
        install_path = input('Enter directory path to install libraries:')
        install_path = Path(install_path).expanduser()
        if not install_path.exists():
            print('Not a vaild install path, enter again.')

    download(download_path)
    install(download_path, build_path, install_path)
