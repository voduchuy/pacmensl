from pathlib import Path
import ext_armadillo
import ext_metis
import ext_parmetis
import ext_zoltan
import ext_petsc
import ext_cvode


def get_lib(download_path, build_path, install_path):
    print('DOWNLOADING THIRD-PARTY LIBRARIES')
    ext_armadillo.download(download_path)
    ext_metis.download(download_path)
    ext_parmetis.download(download_path)
    ext_zoltan.download(download_path)
    ext_petsc.download(download_path)
    ext_cvode.download(download_path)
    print('COMPILING AND INSTALLING THIRD-PARTY LIBRARIES')
    ext_armadillo.install(download_path, build_path, install_path)
    ext_metis.install(download_path, build_path, install_path)
    ext_parmetis.install(download_path, build_path, install_path)
    ext_zoltan.install(download_path, build_path, install_path)
    ext_petsc.install(download_path, build_path, install_path)
    ext_cvode.install(download_path, build_path, install_path)


if __name__ == '__main__':
    try:
        download_path = input('Enter directory path to extract all downloaded source codes:')
        download_path = Path(download_path).expanduser()
        assert(download_path.exists())
    except AssertionError:
        print('Not a valid path, enter again.')

    try:
        build_path = input('Enter directory path to do compilation (must be different from source code directory):')
        build_path = Path(build_path).expanduser()
        assert(build_path.exists())
    except AssertionError:
        print('Not a valid build path, enter again.')

    try:
        install_path = input('Enter directory path to install libraries:')
        install_path = Path(install_path).expanduser()
        assert(install_path.exists())
    except AssertionError:
        print('Not a vaild install path, enter again.')

    get_lib(download_path, build_path, install_path)

