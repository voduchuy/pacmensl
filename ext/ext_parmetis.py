# Script for downloading and installing Parmetis

import wget
import subprocess
from pathlib import Path
import tarfile as tar


def download(path_to):
    dest_dir = Path(path_to)
    print('\n downloading parmetis... ')
    url = 'http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz'
    wget.download(url, str(dest_dir))
    f = tar.open(str(dest_dir/Path('parmetis-4.0.3.tar.gz')))
    f.extractall(dest_dir)


def install(src_path, build_path, install_path):
    src_dir = Path(src_path)/Path('parmetis-4.0.3')
    build_dir = Path(build_path)/Path('parmetis')
    install_dir = Path(install_path)
    src_dir = src_dir.expanduser()
    build_dir = build_dir.expanduser()
    install_dir = install_dir.expanduser()

    if not build_dir.exists():
        build_dir.mkdir()

    print('configure parmetis...')
    subprocess.call([
        'cmake', '-DCMAKE_C_COMPILER=mpicc',
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
        '-DSHARED=1',
        "-DMETIS_PATH="+str(src_dir.resolve()/Path('metis')),
        "-DGKLIB_PATH="+str(src_dir.resolve()/Path('metis/GKlib')),
        '-DCMAKE_INSTALL_PREFIX='+str(install_dir),
        '-DCMAKE_C_FLAGS=-O3',
        str(src_dir.resolve())
    ],
                    cwd=build_dir)
    print('build parmetis...')
    subprocess.call(['make'], cwd=build_dir.resolve())
    print('install parmetis...')
    subprocess.call(['make', 'install'], cwd=build_dir.resolve())


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