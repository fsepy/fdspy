# -*- coding: utf-8 -*-
import json
import os
import re
import subprocess
import sys
import time
import collections
import pandas
from queue import Queue, Empty
from threading import Thread
import copy
import shutil


class FdsJob(object):
    def __init__(
        self,
        uid,
        path_fds,
        num_mpi=1,
        num_omp=1,
        path_work=None,
        user="n/a",
        path_destination=None,
        **red_kwargs,
    ):

        # ASSIGN USER DEFINED PROPERTIES

        self.uid = uid
        if path_work is None:
            self.path_work = os.path.dirname(path_fds)
        self.path_fds = os.path.realpath(path_fds)
        self.path_destination = path_destination
        self.num_mpi = int(num_mpi)
        self.num_omp = int(num_omp)
        # self.config_sp_timeout = config_sp_timeout
        self.user = user

        # ASSIGN DERIVED PROPERTIES

        with open(self.path_fds, "r") as f:
            self._fds_str = f.read()
        # self._chid = re.compile("'[\S]+'").findall()[0].replace("'", '')

        self._chid = (
            re.compile("'[\S]+'")
            .search(
                re.compile("CHID=[\s\S]+?[/,]")
                .search(re.compile("&HEAD[\s\S]*?/").search(self._fds_str).group(0))
                .group(0)
            )
            .group(0)
            .replace("'", "")
        )

        self._uptime = (
            re.compile("[\d.]+")
            .search(
                re.compile("T_END=[\s\S\d.]*?[,/]")
                .search(re.compile("&TIME[\s\S]*?/").search(self._fds_str).group(0))
                .group(0)
            )
            .group(0)
        )

        self._uptime = float(self._uptime)

        self._current_progress = 0

        self.sp = None

    def run_job(self):

        os.chdir(self.path_work)
        os.environ["OMP_NUM_THREADS"] = "{:d}".format(self.num_omp)

        if self.num_mpi == 1:
            str_cmd = "fds {}".format(os.path.basename(self.path_fds))
        else:
            str_cmd = "mpiexec -localonly -n {:d} fds {}".format(
                self.num_mpi, os.path.basename(self.path_fds)
            )

        self.sp = subprocess.Popen(
            args=str_cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self.path_work,
            bufsize=-1,
            close_fds="posix" in sys.builtin_module_names,
        )

    def check_is_live(self):
        if self.sp.poll() is not None:
            return False
        else:
            return True

    def get_current_progress(self):

        path_out_file = os.path.join(self.path_work, "{}.out".format(self._chid))

        try:
            with open(path_out_file, "r") as f:
                str_out_file = f.read()
        except FileNotFoundError:
            str_out_file = ""

        try:
            self._current_progress = float(
                re.compile("[\d.]+").findall(
                    re.compile("Total Time:[\s\S\d.]+?s").findall(str_out_file)[-1]
                )[-1]
            )
        except IndexError:
            pass

        return self._current_progress

    def copy_files(self):
        if self.path_destination:
            shutil.copytree(
                self.path_work,
                os.path.join(self.path_destination, os.path.basename(self.path_work)),
            )


class ClientAgentBack(object):
    def __init__(self, path_fds_queue, available_cpu_cores):

        self.path_fds_queue_json = path_fds_queue
        self.count_cpu_cores_available = available_cpu_cores

        self._count_live_jobs = 0
        self._count_cpu_cores_used = 0
        self._list_FdsJob = []

    def start(self, time_wait=30.0):

        while True:
            self.update_queue()

            # CHECK CPU USAGE
            if (self.count_cpu_cores_available - self._count_cpu_cores_used) <= 0:
                time.sleep(time_wait)
                continue

            # READ FDS QUEUE
            fds_queue_pending = self.queue_status(0)
            if len(fds_queue_pending) == 0:
                time.sleep(time_wait)
                continue

            # CHECK NEXT IN QUEUE
            fds_job_config_key_next = sorted(fds_queue_pending.keys())[0]
            fds_job_config = fds_queue_pending[fds_job_config_key_next]

            # RUN FDS JOB IF ENOUGH RESOURCE
            count_pt = fds_job_config["num_omp"] * fds_job_config["num_mpi"]
            count_cpu_available = (
                self.count_cpu_cores_available - self._count_cpu_cores_used
            )
            if count_cpu_available >= count_pt:
                print("available cpu:", count_cpu_available)
                print("required cpu:", count_pt)
                # QUEUE FDS
                j = FdsJob(uid=fds_job_config_key_next, **fds_job_config)
                j.run_job()
                self._list_FdsJob.append(j)
                self.update_queue(j.uid, status=1)
                print(fds_job_config)

            time.sleep(time_wait)

    def queue_read(self):
        with open(self.path_fds_queue_json, "r") as f:
            j = json.load(f)
        return j

    def update_queue(
        self,
        uid=None,
        path_fds=None,
        num_omp=None,
        num_mpi=None,
        progress=None,
        status=None,
    ):
        j = self.queue_read()

        if uid:
            if path_fds:
                j[uid]["path_fds"] = path_fds
            if num_omp:
                j[uid]["num_omp"] = num_omp
            if num_mpi:
                j[uid]["num_mpi"] = num_mpi
            if progress:
                j[uid]["progress"] = progress
            if status:
                j[uid]["status"] = status

        count_tc = 0
        for i, FdsJob in enumerate(self._list_FdsJob):
            if FdsJob.check_is_live():
                count_tc += FdsJob.num_omp * FdsJob.num_mpi
                j[FdsJob.uid]["progress"] = FdsJob.get_current_progress()
            else:
                j[FdsJob.uid]["status"] = 2
                FdsJob.copy_files()
                del self._list_FdsJob[i]
            j[FdsJob.uid]["progress"] = FdsJob.get_current_progress()

        # UPDATE LIVE JOB COUNT
        self._count_live_jobs = len(self._list_FdsJob)
        self._count_cpu_cores_used = count_tc

        with open(self.path_fds_queue_json, "w") as f:
            json.dump(j, f, indent=4)

    def queue_status(self, status):
        q = self.queue_read()
        q_ = copy.copy(q)
        for k, v in q.items():
            if v["status"] != status:
                del q_[k]

        return q_


class ClientAgentFront(object):
    def __init__(self, path_fds_queue):
        self.path_fds_queue = path_fds_queue

        self.__QUEUE_FORMAT = {
            "path_fds": "FDS INPUT FILE PATH",
            "path_destination": None,
            "num_omp": "NUMBER OF MP THREADS",
            "num_mpi": "NUM OF MPI PROCESS",
            "progress": None,
            "status": None,
        }

    def start(self):
        while True:
            cmd = input(">>>")

            if cmd == "exit":
                break
            elif cmd == "append":
                self.queue_write(self.queue_append())
            elif cmd == "help":
                print("append, delete, insert, ")
            else:
                print("Unknown command.")

    def queue_append(self):
        dict_queue = self.queue_read()
        if len(dict_queue) > 0:
            uid = "{:d}".format(int(list(dict_queue)[-1]) + 1)
        else:
            uid = 0

        dict_queue[uid] = {}
        dict_queue[uid]["path_fds"] = self._input_path("Drop *.fds file: ")
        dict_queue[uid]["num_omp"] = int(input("Number of OMP threads: "))
        dict_queue[uid]["num_mpi"] = int(input("Number of MPI processes: "))
        dict_queue[uid]["progress"] = 0
        dict_queue[uid]["status"] = 0
        dict_queue[uid]["path_destination"] = self._input_path(
            "Drop destination folder: "
        )

        return dict_queue

    def queue_delete(self):
        pass

    def queue_insert(self):
        pass

    def queue_read(self):
        with open(self.path_fds_queue, "r") as f:
            # dict_queue = json.load(f)
            return collections.OrderedDict(sorted(json.load(f).items()))

    def queue_write(self, str):
        with open(self.path_fds_queue, "w") as f:
            json.dump(str, f, indent=4)

    @staticmethod
    def _input_path(msg):
        r = input(msg)
        if r[0] == r[-1] == '"' or r[0] == r[-1] == "'":
            return os.path.realpath(r[1:-1])
        else:
            return os.path.realpath(r)


if __name__ == "__main__":

    # CA = ClientAgentBack(
    #     path_fds_queue=r"C:\Users\ian\Desktop\fdspy_test\fds_batch.json",
    #     available_cpu_cores=4
    # )
    # CA.start(time_wait=1)

    CAF = ClientAgentFront(
        path_fds_queue=r"C:\Users\IanFu\Desktop\fdspy_test\fds_queue.json"
    )
    CAF.start()

    pass
