"""fdspy CLI Help.
Usage:
    fdspy stats <file_name>

Options:
    -h --help       to show help.

Commands:
    fds stats       to analysis FDS input source code
"""

from docopt import docopt


def main():
    import os

    arguments = docopt(__doc__)

    arguments["<file_name>"] = os.path.realpath(arguments["<file_name>"])

    if arguments["stats"]:

        from fdspy.lib.fds2dict import main_cli

        filepath_fds = os.path.realpath(arguments["<file_name>"])

        main_cli(filepath_fds=filepath_fds)
