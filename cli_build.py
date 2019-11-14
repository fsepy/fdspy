# -*- coding: utf-8 -*-
import os
import subprocess


def build_cli():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    cmd_pyinstaller = 'python -m PyInstaller'
    cmd_option_list = [
        '--noconsole',
        '--onefile',
        '--windowed',
    ]
    cmd_extra_files_list = [

    ]
    cmd_script = 'cli.py'

    cmd_options = ' '.join(cmd_option_list)
    cmd_extra_files = ' '.join(cmd_extra_files_list)

    cmd_options = cmd_options + ' ' if len(cmd_options) > 0 else ''
    cmd_extra_files = cmd_extra_files + ' ' if len(cmd_extra_files) > 0 else ''

    cmd = f'{cmd_pyinstaller} {cmd_options}{cmd_extra_files}{cmd_script}'

    print(cmd)
    subprocess.call(cmd)


if __name__ == '__main__':
    build_cli()
