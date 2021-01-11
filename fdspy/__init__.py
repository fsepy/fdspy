from os import path
import logging

__root_dir__ = path.dirname(path.realpath(__file__))


# setup logger
def __get_logger():
    logger_ = logging.getLogger('gui')

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'))
    logger_.addHandler(c_handler)

    logger_.setLevel(logging.INFO)

    return logger_


logger = __get_logger()

"""
VERSION IDENTIFICATION RULES DOCUMENTED IN PEP 440 ARE FOLLOWED.

Version scheme
==============

Distributions are identified by a public version identifier which supports all defined version comparison operations

The version scheme is used both to describe the distribution version provided by a particular distribution archive, as
well as to place constraints on the version of dependencies needed in order to build or run the software.

Public version identifiers
--------------------------

The canonical public version identifiers MUST comply with the following scheme:

`[N!]N(.N)*[{a|b|rc}N][.postN][.devN]`

Public version identifiers MUST NOT include leading or trailing whitespace.

Public version identifiers MUST be unique within a given distribution.

See also Appendix B : Parsing version strings with regular expressions which provides a regular expression to check
strict conformance with the canonical format, as well as a more permissive regular expression accepting inputs that may
require subsequent normalization.

Public version identifiers are separated into up to five segments:

    - Epoch segment: N!
    - Release segment: N(.N)*
    - Pre-release segment: {a|b|rc}N
    - Post-release segment: .postN
    - Development release segment: .devN

"""

__version__ = "0.0.2"


if __name__ == "__main__":
    import re

    def is_canonical(version):
        return (
            re.match(
                r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
                version,
            )
            is not None
        )

    assert is_canonical(__version__)
