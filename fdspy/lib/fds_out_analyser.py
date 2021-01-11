import re
from abc import ABC
from datetime import datetime
from os.path import join, dirname

from fdspy import logger


class FDSOutBaseModel(ABC):
    def __init__(self):
        self.__fds_out_raw: str = None
        self.__fds_out_raw_changed: bool = True

        self.__simulation_time_stats: list = None

    def read_fds_out(self, fds_out_raw: str):
        self.fds_out_raw = fds_out_raw

    @staticmethod
    def __make_simulation_time_stats(fds_out_raw: str) -> tuple:

        time_step_data = re.findall(r"Time Step[ \S]*?\n[ \S]+?Step Size:[ \S]+?\n", fds_out_raw)

        time_step = list()
        simulation_time = list()
        date = list()
        wall_time_elapse = list()
        for i in time_step_data:
            try:
                _ = re.findall(r'Time Step *([0-9]+)', i)[0]
                time_step.append(_)
            except Exception as e:
                print(e)
            try:
                _total_time_str = re.findall(r'Total Time: *([.0-9]*) *s', i)[0]
                _total_time = float(_total_time_str)
            except Exception as e:
                logger.warning(e)
                continue
            try:
                _datetime_str = re.findall(
                    r'January[ \S]+\n|February[ \S]+\n|March[ \S]+\n|April[ \S]+\n|May[ \S]+\n|June[ \S]+\n|'
                    r'July[ \S]+\n|August[ \S]+\n|September[ \S]+\n|October[ \S]+\n|November[ \S]+\n|December[ \S]+\n', i
                )[0].strip()
                _datetime_cls = datetime.strptime(_datetime_str, '%B %d, %Y  %H:%M:%S')
            except Exception as e:
                logger.warning(e)
                continue

            simulation_time.append(_total_time)
            date.append(_datetime_cls)
            try:
                wall_time_elapse.append((_datetime_cls - date[0]).total_seconds())
            except IndexError:
                wall_time_elapse.append((_datetime_cls - _datetime_cls).total_seconds())

        return time_step, simulation_time, wall_time_elapse

    @property
    def fds_out_raw(self) -> str:
        return self.__fds_out_raw

    @fds_out_raw.setter
    def fds_out_raw(self, fds_out_raw: str):
        self.__fds_out_raw = fds_out_raw
        self.__fds_out_raw_changed = True

    def make_simulation_time_stats(self, fp_csv: str = None):
        """
        Make simulation time stats.
        A list consists of the following:
            time_step, total_time, wall_time_elapse
        """

        def write_csv(fp_csv_, time_step_, simulation_time_, wall_time_elapse_):
            l0 = ['time_step']
            l0.extend(time_step_)
            l1 = ['simulation_time']
            l1.extend(simulation_time_)
            l2 = ['wall_time_elapse']
            l2.extend(wall_time_elapse_)
            with open(fp_csv_, 'w+') as f_:
                f_.write('\n'.join(['{},{},{}'.format(*i) for i in list(zip(l0, l1, l2))]))

        # Make simulation time stats
        # only process data if (*.out data has changed) or (no previous simulation time stats have been made)
        if self.__fds_out_raw_changed or self.__simulation_time_stats is None:
            # raise error if no *.out data is defined
            if self.fds_out_raw is None:
                raise ValueError(f'FDS output data is undefined, {self.fds_out_raw}')
            else:
                # process *.out data and make simulation time stats
                time_step, simulation_time, wall_time_elapse = self.__make_simulation_time_stats(self.fds_out_raw)
                self.__simulation_time_stats = dict(
                    time_step=time_step,
                    simulation_time=simulation_time,
                    wall_time_elapse=wall_time_elapse,
                )
                pass
        elif self.__simulation_time_stats is not None:
            pass
        else:
            raise ValueError('Unforeseen condition')

        # Write to csv (optional)
        if fp_csv is not None:
            write_csv(
                fp_csv_=fp_csv,
                time_step_=self.__simulation_time_stats['time_step'],
                simulation_time_=self.__simulation_time_stats['simulation_time'],
                wall_time_elapse_=self.__simulation_time_stats['wall_time_elapse'],
            )

        return self.__simulation_time_stats


def _test_make_simulation_time_stats():
    """
    Test function for `FDSOutBaseModel.make_simulation_time_stats`
    """
    from fdspy.tests.fds_out import general_benchmark_1

    model = FDSOutBaseModel()  # instantiate class
    model.read_fds_out(general_benchmark_1)  # assign *.out data
    stats = model.make_simulation_time_stats()  # make simulation time stats

    # check `FDSOutBaseModel.make_simulation_time_stats` returns the expected dict
    assert all([i in stats for i in ['time_step', 'simulation_time', 'wall_time_elapse']])

    # check no. of time records, should be 29
    assert all([len(stats[i]) == 29 for i in ['time_step', 'simulation_time', 'wall_time_elapse']])


if __name__ == '__main__':

    _test_make_simulation_time_stats()

    import os.path as path

    fp_out = path.realpath(r'C:\Users\IanFu\Desktop\CLT-S9-P0_MLR_3.out')

    with open(fp_out, 'r') as f:
        fds_out_raw_ = f.read()

    from fdspy.tests.fds_out import general_benchmark_1

    model = FDSOutBaseModel()
    model.read_fds_out(fds_out_raw_)

    fp_csv = join(dirname(fp_out), 'test_out_simulation_time.csv')
    model.make_simulation_time_stats(fp_csv=fp_csv)