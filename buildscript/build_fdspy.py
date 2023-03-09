import os
import subprocess
import sys
from os.path import dirname, exists, realpath, join


def build_gui(app_name: str, fp_target_py: str, options: list = None):
    print('\n' * 2)

    os.chdir(dirname(realpath(__file__)))

    cmd_option_list = [
        f'-n={app_name}',
        '--exclude-module=docopt',
        '--exclude-module=setuptools',
        '--exclude-module=ipython',
        '--exclude-module=jedi',
        '--exclude-module=cython',
    ]

    cmd_option_list.extend([
        '--exclude-module=tk',
        '--exclude-module=_tkinter',
        '--exclude-module=tkinter',
    ])

    if options:
        cmd_option_list.extend(options)

    cmd = ['pyinstaller'] + cmd_option_list + [fp_target_py]
    print(f'COMMAND: {" ".join(cmd)}')

    with open('pyinstaller_build.log', 'wb') as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for c in iter(lambda: process.stdout.read(1), b''):  # replace '' with b'' for Python 3
            sys.stdout.write(c.decode('utf-8'))
            f.write(c)


def find_fp(dir_work: str, endswith: tuple = None) -> list:
    list_fp = list()
    list_fp_append = list_fp.append

    for dirpath, dirnames, filenames in os.walk(dir_work):

        for fn in filenames:
            if endswith is None:
                list_fp_append(join(dirpath, fn))
            elif any([fn.endswith(suffix) for suffix in endswith]):
                list_fp_append(join(dirpath, fn))

    return list_fp


def main():
    options = [
        "--onefile",  # output unpacked dist to one directory, including an .exe file
        "--noconfirm",  # replace output directory without asking for confirmation
        "--clean",  # clean pyinstaller cache and remove temporary files
    ]

    # add upx if it exists in the buildscript folder
    if sys.platform.startswith('win') and exists(f"--upx-dir={join(dirname(__file__), 'upx.exe')}"):
        options += [f"--upx-dir={join(dirname(__file__), 'upx.exe')}"]

    build_gui('FDSPY', options=options, fp_target_py=os.path.join(dirname(dirname(__file__)), 'fdspy', 'cli.py'))


if __name__ == "__main__":
    main()
