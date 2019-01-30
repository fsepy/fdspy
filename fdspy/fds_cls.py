# -*- coding: utf-8 -*-
import json
import os
import re
import subprocess
import sys
import time
from queue import Queue, Empty
from threading import Thread


class FdsJob:
    def __init__(
            self, uid, path_fds, num_mpi=1, num_omp=1, path_work=None, config_sp_timeout=30 * 24 * 60 * 60,
            user='n/a'
    ):

        # ASSIGN USER DEFINED PROPERTIES

        self.uid = uid
        if path_work is None:
            self.path_work = os.path.dirname(path_fds)
        self.path_fds = os.path.realpath(path_fds)
        self.num_mpi = int(num_mpi)
        self.num_omp = int(num_omp)
        self.config_sp_timeout = config_sp_timeout
        self.user = user

        # ASSIGN DERIVED PROPERTIES

        self._rep_simulation_time_1 = re.compile('Simulation Time:[\s.=0-9]*')
        self._rep_simulation_time_2 = re.compile('\d+\.\d*')

        self._current_progress = 0
        self._is_live = False
        self._time_started = None
        self._time_finished = None
        # self.std_out = None

        self.sp = None
        self.sp_queue = None
        self.sp_threads = None

    @staticmethod
    def enqueue_output(stdout, queue):
        for line in iter(stdout.readline, b''):
            queue.put(line)
        stdout.close()

    def run_job(self):

        # Set properties
        self._time_started = time.time()
        self._is_live = True

        # Change work directory to the *.fds file's location
        os.chdir(self.path_work)

        # Set number of Threads
        os.environ['OMP_NUM_THREADS'] = '{:d}'.format(self.num_omp)

        # Make run fds command
        if self.num_mpi == 1:
            str_cmd = 'fds {}'.format(os.path.basename(self.path_fds))
        else:
            str_cmd = 'mpiexec -localonly -n {:d} fds {}'.format(self.num_mpi, os.path.basename(self.path_fds))

        # Run FDS, summon subprocess
        self.sp = subprocess.Popen(
            args=str_cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self.path_work,
            bufsize=-1,
            close_fds='posix' in sys.builtin_module_names
        )

        self.sp_queue = Queue()
        self.sp_threads = Thread(
            target=self.enqueue_output, args=(self.sp.stdout, self.sp_queue)
        )
        self.sp_threads.daemon = True  # thread dies with the program
        self.sp_threads.start()

    def get_stdout_line(self, timeout=1):

        # read line without blocking
        try:
            line = self.sp_queue.get(timeout=timeout)  # or q.get_nowait()
            try:
                stdout_line = line.decode('utf-8')
            except IndexError:
                stdout_line = None

        except Empty:
            stdout_line = None

        return stdout_line

    def get_current_progress(self, timeout=1):

        if self.check_is_live():
            pg = self.get_stdout_line(timeout)
            # print(type(pg))
            # print(pg)
            # pg = pg.decode('utf-8')
            try:
                pg = float(self._rep_simulation_time_2.findall(self._rep_simulation_time_1.findall(pg)[-1])[0])
            except (IndexError, TypeError):
                pg = 0
        else:
            pg = 0

        self._current_progress = max(pg, self._current_progress)

        return self._current_progress

    def check_is_live(self):
        if self.sp.poll() is not None:
            return False
        else:
            return True


class UserClientBG:
    def __init__(self, path_fds_queue, available_cpu_cores, path_current_summary):

        self.path_fds_queue = path_fds_queue
        self.path_current_summary = path_current_summary
        self.count_cpu_cores_available = available_cpu_cores

        self._count_live_jobs = 0
        self._count_cpu_cores_used = 0
        self._list_FdsJob = []
        self._dict_all_jobs = {"-1": {
            "USER": "Example",
            "JOB NAME": "example.fds",
            "CPU CORES": 2,
            "PROGRESS": 100,
            "STATUS": "COMPLETE"
        }}

    def start(self, time_wait=30):

        while True:
            # UPDATE STATUS
            self.status_update()

            # print("live jobs count:", self._count_live_jobs)
            # print("cpu used:", self._count_cpu_cores_used)

            # CHECK AVAILABLE CPU USAGE
            if (self.count_cpu_cores_available - self._count_cpu_cores_used) <= 0:
                time.sleep(time_wait)
                continue

            # READ FDS QUEUE
            fds_queue = self.queue_read(self.path_fds_queue)
            if len(fds_queue) == 0:
                time.sleep(time_wait)
                continue

            # CHECK NEXT IN QUEUE
            fds_job_config_key_next = sorted(fds_queue.keys())[0]
            fds_job_config = fds_queue[fds_job_config_key_next]
            num_tc = fds_job_config['num_mpi'] * fds_job_config['num_omp']

            # RUN FDS JOB IF ENOUGH RESOURCE
            if self.count_cpu_cores_available - self._count_cpu_cores_used >= num_tc:
                # QUEUE FDS
                if False:
                    print(fds_job_config)
                    self.queue_delete(self.path_fds_queue, fds_job_config_key_next)
                    continue
                j = FdsJob(uid=fds_job_config_key_next, **fds_job_config)
                j.run_job()
                self._list_FdsJob.append(j)
                self._dict_all_jobs[fds_job_config_key_next] = {
                    "USER": "N/A",  # todo
                    "JOB NAME": os.path.basename(fds_job_config["path_fds"]),
                    "PROGRESS": 0,
                    "CPU CORES": num_tc,
                    "STATUS": 'Live',
                }
                print('running {}'.format(fds_job_config))
                # DELETE QUEUE AND WRITE NEW QUEUE FILE
                self.queue_delete(self.path_fds_queue, fds_job_config_key_next)

            time.sleep(time_wait)

    @staticmethod
    # ABLE TO VIEW QUEUED TASKS
    def queue_read(path_fds_queue):
        with open(path_fds_queue, 'r') as f:
            fds_queue = json.load(f)

        time.sleep(0.2)

        return fds_queue

    @staticmethod
    def queue_delete(path_fds_queue, key):
        with open(path_fds_queue, 'r') as f:
            fds_queue = json.load(f)
        del fds_queue[key]
        with open(path_fds_queue, 'w') as f:
            json.dump(fds_queue, f, indent=4, sort_keys=True)
        return fds_queue

    def status_update(self):
        # UPDATE LIVE JOB LIST

        count_tc = 0

        for i, FdsJob in enumerate(self._list_FdsJob):
            if FdsJob.check_is_live():
                FdsJob.get_current_progress(timeout=5)
                count_tc += (FdsJob.num_omp * FdsJob.num_mpi)
            else:
                self._dict_all_jobs[FdsJob.uid]["STATUS"] = "Complete"
                del self._list_FdsJob[i]

        # UPDATE LIVE JOB COUNT
        self._count_live_jobs = len(self._list_FdsJob)
        self._count_cpu_cores_used = count_tc

    def get_current_live_job_summary(self):
        # USER, FILE NAME, PROGRESS, IS COMPLETE, CPU CORES

        # dict_data = {
        #     "FILE NAME": [],
        #     "PROGRESS": [],
        #     "CPU CORES": [],
        #     "STATUS": [],
        # }

        for i, FdsJob in enumerate(self._list_FdsJob):
            self._dict_all_jobs[FdsJob.uid]["PROGRESS"] = FdsJob._current_progress
        pass


if __name__ == '__main__':

    C = UserClientBG(
        path_fds_queue=r"C:\Users\ian\Desktop\fdspy_test\fds_batch.json",
        path_current_summary=r"C:\Users\ian\Desktop\fdspy_test\current_summary.csv",
        available_cpu_cores=4
    )

    C.start(time_wait=1)

    # import collections
    #
    # while True:
    #
    #     with open(r"C:\Users\ian\Desktop\fdspy_test\fds_batch.json", 'r') as f:
    #         fds_queue = json.load(f)
    #
    #     fds_queue = collections.OrderedDict(sorted(fds_queue.items()))
    #     list_fds_queue_keys = list(fds_queue)
    #
    #     queue_last = int(list_fds_queue_keys[-1])
    #     queue_length = len(list_fds_queue_keys)
    #
    #     fds_job_config = fds_queue[list_fds_queue_keys[0]]
    #     del fds_queue[list_fds_queue_keys[0]]
    #
    #     # QUEUE FDS
    #     j = FdsJob(**fds_job_config)
    #     j.run_job()
    #
    #     while True:
    #
    #         if j.sp.poll() is not None:
    #             break
    #         else:
    #             print(j.get_stdout_line())
    #
    #     with open(r"C:\Users\ian\Desktop\fdspy_test\fds_batch.json", 'w') as f:
    #         json.dump(fds_queue, f, indent=4, sort_keys=True)
    #


    pass
