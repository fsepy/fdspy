if __name__ == "__main__":

    from fdspy.fds_cls import ClientAgentBack
    from fdspy.func import input_path
    import os

    os.system("title " + "FDS BACK")

    print("Background agent is running...")

    CAB = ClientAgentBack(
        path_fds_queue=os.path.join("C:", os.sep, "APP", "fdspy", "fds_queue.json"),
        available_cpu_cores=int(input_path("Available cpu cores: ")),
    )

    CAB.start(time_wait=20)
