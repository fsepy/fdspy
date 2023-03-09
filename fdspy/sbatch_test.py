from os.path import dirname, join

from sbatch import *


def test_make_sh():
    filepath_fds = dirname(dirname(dirname(__file__)))
    filepath_fds = join(filepath_fds, 'tests_fds', 'residential_corridor.fds')

    sh = make_sh(filepath_fds=filepath_fds)

    print(sh)


if __name__ == '__main__':
    test_make_sh()
