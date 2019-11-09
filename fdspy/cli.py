"""fdspy CLI Help.
Usage:
    fdspy stats
    fdspy stats <file_name>

Options:
    -h --help       to show help.

Commands:
    fdspy stats     to analysis all .fds files in the current working directory
"""

from docopt import docopt
import plotly


def main():
    import os

    arguments = docopt(__doc__)

    if arguments["stats"]:

        from fdspy.lib.fds_script_proc_analyser import main_cli

        if arguments["<file_name>"]:
            filepath_fds = os.path.realpath(arguments["<file_name>"])
            dict_out = main_cli(filepath_fds=filepath_fds)

            print(dict_out["str"])

            with open(filepath_fds + ".stats.txt", "w+") as f:
                f.write(dict_out["str"])

            plotly.io.write_html(
                dict_out["fig_hrr"],
                file=filepath_fds + ".hrr.html",
                auto_open=True,
                config={
                    "scrollZoom": False,
                    "displayModeBar": True,
                    "editable": True,
                    "showLink": False,
                    "displaylogo": False,
                },
            )

        else:
            fn = list()
            for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
                fn.extend(filenames)
                break
            for i in fn:
                if i.endswith(".fds"):
                    try:
                        dict_out = main_cli(filepath_fds=i)

                        print(dict_out["str"])

                        with open(i + ".stats.txt", "w+") as f:
                            f.write(dict_out["str"])

                        plotly.io.write_html(
                            dict_out["fig_hrr"],
                            file=i + ".hrr.html",
                            auto_open=False,
                            config={
                                "scrollZoom": False,
                                "displayModeBar": True,
                                "editable": True,
                                "showLink": False,
                                "displaylogo": False,
                            },
                        )
                    except Exception as e:
                        # Just print(e) is cleaner and more likely what you want,
                        # but if you insist on printing message specifically whenever possible...
                        if hasattr(e, "message"):
                            print(e.message)
                        else:
                            print(e)
