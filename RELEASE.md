# Release

**Checklist before releasing**:

- Run tests in `tests` directory.
- Version is up to date in `fdspy.__init__:__version__`.

**Release indicators**:

- New: to indicate a new feature being implemented.
- Improved: to indicate added/improved functionality of an existing feature.
- Fixed: to indicate a fix.
- Depreciated: to indicate something is deleted for good.

## Known issues

- To implement VENT statistics.

## Version history

### 11/11/2019 VERSION 0.0.20:

- Improved: `fdspy.lib.fds_script_proc_analyser:fds_analyser_mesh` added D\*/dx calculation for individual mesh.
- Fixed: `fdspy.lib.fds_script_proc_analyser:fds_analyser_mesh` fixed mesh size calculation.

### 09/11/2019 VERSION 0.0.19: 

- New: `fdspy stats` to analyse all .fds files within a directory and produces stats and HRR plot.
- New: `fdspy.lib.fds_script_proc_decoder` to parameterise raw FDS code into a pandas.DataFrame object.
- New: `fdspy.lib.fds_script_proc_analyser` to analyse FDS code and return some indicative statistics.
- New: `fdspy.lib.fds_script_proc_data` stored some FDS code for initial testing and FDS manual latex for other uses.

### 04/02/2019 VERSION: 0.0.5

- Initial, intended to create a library for handling FDS simulation jobs.