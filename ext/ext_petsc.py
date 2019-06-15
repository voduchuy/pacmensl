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
    src_dir = src_dir.expanduser()
    build_dir = build_dir.expanduser()
    install_dir = install_dir.expanduser()
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
    download_dir = '/Users/huyvo/Codes/software/src/'
    build_dir = '~/Codes/software/build/'
    install_dir = '/Users/huyvo/Codes/software/install/'
    # download(download_dir)
    install(download_dir, build_dir, install_dir)
