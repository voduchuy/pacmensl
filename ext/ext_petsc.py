# Functions for downloading, building and installing armadillo

import wget
import subprocess
from pathlib import Path
import tarfile as tar


def download(path_to):
    dest_dir = Path(path_to)
    dest_dir = dest_dir.expanduser()
    print('cloning petsc... ')
    subprocess.call(['git', 'clone', '-b', 'maint', 'https://bitbucket.org/petsc/petsc', 'petsc'], cwd=dest_dir)


def install(src_path, build_path, install_path):
    src_dir = Path(src_path) / Path('petsc')
    build_dir = Path(build_path) / Path('petsc')
    install_dir = Path(install_path)
    src_dir = src_dir.expanduser().resolve()
    build_dir = build_dir.expanduser().resolve()
    install_dir = install_dir.expanduser().resolve()
    if not build_dir.exists():
        build_dir.mkdir()
    subprocess.call(
        [
            './configure',
            'PETSC_DIR='+str(src_dir),
            '--prefix=' + str(install_dir),
            '--with-precision=double',
            '—-with-threadsafety=1',
            '--with-scalar-type=real',
            '--with-debugging=0',
            'COPTFLAGS=-O2',
            '--with-avx512-kernels=1',
            '--with-shared-libraries=1'
        ],
        cwd=src_dir
    )
    subprocess.call(
        [
            'make', 'PETSC_DIR=' + str(src_dir)
        ],
        cwd=src_dir
    )
    subprocess.call(
        [
            'make', 'PETSC_DIR=' + str(src_dir), 'install'
        ],
        cwd=src_dir
    )


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
