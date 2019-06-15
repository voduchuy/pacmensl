# Functions for downloading, building and installing armadillo

import wget
import subprocess
from pathlib import Path
import tarfile as tar


def download(path_to):
    dest_dir = Path(path_to)
    dest_dir = dest_dir.expanduser()
    print('cloning trilinos... ')
    url = 'https://github.com/trilinos/Trilinos.git'
    subprocess.call(['git', 'clone', url], cwd=dest_dir)


def install(src_path, build_path, install_path):
    src_dir = Path(src_path) / Path('Trilinos')
    build_dir = Path(build_path) / Path('zoltan')
    install_dir = Path(install_path)
    src_dir = src_dir.expanduser()
    build_dir = build_dir.expanduser()
    install_dir = install_dir.expanduser()
    if not build_dir.exists():
        build_dir.mkdir()

    subprocess.call(['cmake', '-DPTL_ENABLE_MPI=ON', '-DTrilinos_ENABLE_Zoltan=ON',
                     '-DCMAKE_INSTALL_PREFIX='+str(install_dir),
                     '-DBUILD_SHARED_LIBS=ON',
                     '-DTPL_ENABLE_ParMETIS=ON',
                     '-DParMETIS_INCLUDE_DIRS='+str(install_dir/Path('include')),
                     '-DParMETIS_LIBRARY_DIRS='+str(install_dir/Path('lib')),
                     '-DTrilinos_GENERATE_REPO_VERSION_FILE=OFF',
                     src_dir], cwd=build_dir)
    subprocess.call(['make', '-j4', 'install'], cwd=build_dir)


if __name__ == "__main__":
    download_dir = '/Users/huyvo/Codes/software/src/'
    build_dir = '~/Codes/software/build/'
    install_dir = '/Users/huyvo/Codes/software/install/'
    # download(download_dir)
    install(download_dir, build_dir, install_dir)
