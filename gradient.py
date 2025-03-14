import ast
import logging
import os
import queue
import random
import subprocess
import threading
import time
from time import sleep

import quilt
import maya
import numpy as np
import pandas as pd
import requests
import toml

from train import Jobstats

# These sensitive environment variables are in .env
from settings import HEADERS, REGISTRY_USER, REGISTRY_PASSWORD

# Load non-sensitive environment variables
config = toml.load("./config.toml")

API_DOMAIN = config["paperspace"]["api_domain"]
LOG_DOMAIN = config["paperspace"]["log_domain"]

#logging.config.fileConfig("./logging.ini", disable_existing_loggers=False)
#logger = logging.getLogger(__name__)


def run_parallel_training(
    executable,
    ps_command_1,
    ps_command_2,
    ps_command_3,
    experiment_name,
    py_script,
    prev_groups,
    opts,
    project_id,
    zone,
    machinetype,
    docker_url,
    image,
    tag,
    workspace,
    ignore_files,
    num_jobs=None,
    FUZZ=False,
):
    def fuzz():
        if FUZZ:
            time.sleep(random.random())

    job_exited = ""

    training_queue = queue.Queue()

    def training_manager():
        """I have EXCLUSIVE RIGHTS to update the job_exited variable"""
        global job_exited

        while True:
            job_exited = training_queue.get()
            fuzz()
            print_queue.put([job_exited, "---------------"])
            fuzz()
            training_queue.task_done()

        # End of training_manager()

    t = threading.Thread(target=training_manager)
    t.daemon = True
    t.start()
    del t

    print_queue = queue.Queue()

    def print_manager():
        """I have EXCLUSIVE RIGHTS to call the "print" keyword"""
        while True:
            job = print_queue.get()
            fuzz()
            for line in job:
                print(line, end="")
                fuzz()
                print()
                fuzz()
            print_queue.task_done()
            fuzz()

        # End of print_manager

    t = threading.Thread(target=print_manager)
    t.daemon = True
    t.start()
    del t

    def worker(
        executable_w,
        ps_command_1_w,
        ps_command_2_w,
        ps_command_3_w,
        experiment_name_w,
        py_script_w,
        ids_w,
        opts_w,
        project_id_w,
        machinetype_w,
        docker_url_w,
        image_w,
        tag_w,
        workspace_w,
        ignore_files_w
    ):
        """My job is to train groups of devices and track which have finished"""
        train_results = train_group(
            executable_w,
            ps_command_1_w,
            ps_command_2_w,
            ps_command_3_w,
            experiment_name_w,
            py_script_w,
            ids_w,
            opts_w,
            project_id_w,
            machinetype_w,
            docker_url_w,
            image_w,
            tag_w,
            workspace_w,
            ignore_files_w
        )
        train_results_err = train_results.stderr.decode("utf-8")
        if train_results_err is not None:
            print('error is ', train_results_err)
            train_results_str = train_results_err
        else:
            train_results_str = train_results.stdout.decode("utf-8")  # NEW
        exit_and_job = get_exit_and_jobid(train_results_str)
        training_queue.put(exit_and_job)
        fuzz()

        # End of worker()

    print_queue.put(["Starting up"])
    fuzz()

    worker_threads = []

    # avoid the repetition of asking for completed ids
    new_groups = devs_to_complete(prev_groups, "tlproject", zone, num_jobs=num_jobs)

    for i, ids in new_groups.items():  # new
        t = threading.Thread(
            target=worker,
            args=(
                executable,
                ps_command_1,
                ps_command_2,
                ps_command_3,
                experiment_name,
                py_script,
                ids,
                opts,
                project_id,
                machinetype,
                docker_url,
                image,
                tag,
                workspace,
                ignore_files
            ),
        )
        worker_threads.append(t)
        t.start()
        fuzz()
    for t in worker_threads:
        fuzz()
        t.join()

    training_queue.join()
    fuzz()
    print_queue.put(["Finishing up!"])
    fuzz()
    print_queue.join()
    fuzz()


def train_group(
    executable,
    ps_command_1,
    ps_command_2,
    ps_command_3,
    experiment_name,
    py_script,
    ids,
    opts,
    project_id,
    machine,
    docker_url,
    image,
    tag,
    workspace=".",
    ignore_files=None
):
    script_path = os.path.join(os.getcwd(), py_script)
    if not os.path.isfile(script_path):
        raise ValueError("cd into the workspace directory.")
    docker_user_url = docker_url + "/" + REGISTRY_USER
    image_and_tag = os.path.join(docker_user_url, image + ":" + tag)
    py_command_and_args = " ".join(["python", py_script, '"' + ids + '"'])
    #py_script_and_args = " ".join(["python", py_script, ids])
    #py_script_and_args = " ".join([py_script_and_args, opts])
    py_script_and_args = " ".join([py_command_and_args, opts])
    print(f"py_script and args: {py_script_and_args}")
    #py_command_and_args = '"' + py_script_and_args + '"'
    py_command_and_args = "'" + py_script_and_args + "'"
    print(f"py_command and args: {py_command_and_args}")
    path_to_executable = '/Users/everyday/miniconda3/envs/ppsquiltthree'
    full_path_to_executable = os.path.join(path_to_executable,
                                           'bin', executable)
    container_subprocess = [
            full_path_to_executable,
            ps_command_1, ps_command_2, ps_command_3,
            "--name", experiment_name,
            "--projectId", project_id,
            "--container", image_and_tag,
            "--machineType", machine,
            "--workspace", workspace]

    if ignore_files is not None:
        container_subprocess.extend(["--ignoreFiles", ignore_files])

    container_subprocess.extend(["--command", py_command_and_args,
                                 "--registryUsername", REGISTRY_USER,
                                 "--registryPassword", REGISTRY_PASSWORD])

    container_str = " ".join(container_subprocess)
    for elem in container_subprocess:
        print(elem)
    print(f"Container subprocess {container_str}")
    container_output = subprocess.run(
        container_subprocess,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return container_output

    # End of train_group


def get_exit_and_jobid(log_str):
    print("log_str in get_exit_and_jobid: ", log_str)
    if "Job Stopped" in log_str:
        print(f"Job stopped: log_str is {log_str}")
        exitcode = get_content_of_key(
            log_str, "Job Stopped", "exitCode", delim=",", is_int=True
        )
    else:
        exitcode = "NA"
    if "New jobId" in log_str:
        jobid = log_str.split("New jobId: ")[-1]
    else:
        print(f"New jobId not in log_str: log_str is: {log_str}")
        jobid = get_content_of_key(log_str, "New", "jobId", delim=",")

    exit_job = "exitCode {}, jobId {}".format(exitcode, jobid)
    return exit_job


def get_content_of_key(content, preceding, key, delim=None, is_int=False):
    print(f"Content: {content}")
    after = content.split(preceding)[-1]
    if delim is not None:
        after = after.strip(delim)
    for t in [" ", ":"]:
        after = after.strip(t)
    before_newline = after.split("\n")[0]
    print(f"Value of before_newline: {before_newline}")
    failed = [a for a in after.split("\n") if a[:10] == "Job Failed"]
    if len(failed) == 1:
        exit_code = failed.split()[-1]
        v = exit_code
    else:
        try:
            k, v = before_newline.split(" ")
            k_clean = k.strip(":")
            if k_clean != key:
                raise ValueError("{} does not match {}".format(k_clean, key))
        except ValueError as e:
            print(f"Value of before_newline: {before_newline}")
            print(e)
    if is_int:
        return int(v)
    else:
        return v


def devs_to_complete(prev_groups, project, zone, num_jobs=None):
    if num_jobs == 0:
        completed = []
    elif num_jobs is None:
        num_jobs = len(prev_groups)
        assert num_jobs > 0
    if num_jobs != 0:
        completed = ids_trained_prev_jobs(project, zone, num_jobs=num_jobs)

    #all_ids_still_to_complete = set([ids for ids in prev_groups.values()])
    all_ids_still_to_complete = set()
    for ids in prev_groups.values():
        print(ids.split(','))
        print(type(ids.split(',')))
        all_ids_still_to_complete.update(ids.split(','))
    ids_to_complete = []
    new_groups = {}
    for k, v in prev_groups.items():
        devs_in_group = v.split(",")
        to_complete_in_group = set(devs_in_group) - set(completed)
        if to_complete_in_group:
            if len(to_complete_in_group) <= 5:
                ids_to_complete.extend(to_complete_in_group)
            else:
                new_groups[k] = ",".join(to_complete_in_group)

    num_to_complete = len(ids_to_complete)
    all_ids_grouped_to_complete = set()
    if num_to_complete > 0:
        group_id = max(prev_groups.keys()) + 1
        begin_idx = 0
        while num_to_complete > 0:
            devs_to_group = min(num_to_complete, 12)
            devs = ids_to_complete[begin_idx : begin_idx + devs_to_group]
            new_groups[group_id] = ",".join(devs)
            all_ids_grouped_to_complete.update(devs)
            num_to_complete -= devs_to_group
            group_id += 1
            begin_idx += devs_to_group
        #all_ids_grouped_to_complete = set([ids for ids in new_groups.values()])
        print('all ids grouped', all_ids_grouped_to_complete)
        print('all ids still to complete', all_ids_still_to_complete)
        assert all_ids_grouped_to_complete == all_ids_still_to_complete
    return new_groups


def ids_trained_prev_jobs(project, zone, num_jobs=5):
    if num_jobs == 0:
        return dict([])

    last_jobs_run = last_jobs(project=project, zone=zone, num_jobs=num_jobs)
    if last_jobs_run.empty:
        print("last_run_run is empty")
        return dict([])
    else:
        completed_ids = []
        job_ids = last_jobs_run["job_id"].values
        for job in job_ids:
            completed_ids.extend(last_ids_trained(job))
        completed_devs_jobs = dict(completed_ids)
        return completed_devs_jobs


def last_jobs(
    num_jobs=None, exit_codes="0,137,255", project=None, zone=None,
):
    jobslist = _list_jobs(project)
    if not jobslist or num_jobs == 0:
        return []
    else:
        jobs_running = [j["id"] for j in jobslist if j["state"] == "running"]
    if jobs_running:
        job_df = pd.DataFrame()
    else:
        # Code 137: container was killed
        if exit_codes is not None:
            exits = [int(c) for c in exit_codes.split(",")]
            finished_jobs = [
                j["id"]
                for j in jobslist
                if j["dtTeardownFinished"] is not None and j["exitCode"] in exits
            ]
        else:
            finished_jobs = [
                j["id"] for j in jobslist if j["dtTeardownFinished"] is not None
            ]

        teardown_times = [
            np_dt_in_zone(j["dtTeardownFinished"], zone=zone)
            for j in jobslist
            if j["id"] in finished_jobs
        ]

        exit_codes = [int(j["exitCode"]) for j in jobslist if j["id"] in finished_jobs]

        job_meta = {
            "teardown_times": teardown_times,
            "exit_codes": exit_codes,
            "job_id": finished_jobs,
        }
        job_df = pd.DataFrame.from_dict(job_meta)
        job_df.sort_values("teardown_times", ascending=False, inplace=True)
    if num_jobs is not None:
        jobs = pd.DataFrame(job_df.iloc[:num_jobs])
        assert len(jobs) == num_jobs
        return jobs
    else:
        return job_df


def np_dt_in_zone(x, zone=None):
    return np.datetime64(maya.parse(x).datetime(to_timezone=zone, naive=True))


def _list_jobs(project):
    jobs_list = "/".join(["jobs", "getJobs"])
    payload = {"project": project}
    listjobs_call = "/".join([API_DOMAIN, jobs_list])

    r = requests.get(listjobs_call, params=payload, headers=HEADERS)
    response = r.json()
    if response:
        return response
    else:
        return []


def last_ids_trained(
    jobid, train_func="profile_training_for_id", log_desc="training time dict for id",
):
    payload = {"jobId": jobid}
    log_call = "/".join([LOG_DOMAIN, "jobs", "logs"])
    r = requests.get(log_call, params=payload, headers=HEADERS)
    delayed_attempts = 0
    while r.status_code == 401:
        if delayed_attempts == 4:
            print("Attempt 4: status code of last_ids_trained: ", r.status_code)
            print("type of status code:", type(r.status_code))
            print("r.text:", r.text)
            raise Exception("Made {} attempts".format(delayed_attempts))
        else:
            sleep(30)
            r = requests.get(log_call, params=payload, headers=HEADERS)
            delayed_attempts += 1

    log_content = r.json()

    end_str = log_desc

    dev_ids_jobs = []
    for logged in log_content:
        if end_str in logged["message"]:
            preamble, train_profile = logged["message"].split(":{")
            dev_id = preamble.split()[-1]
            dev_ids_jobs.append((dev_id, jobid))
    return dev_ids_jobs


def profile_ex_creation(log_file, input_metas, begin_str, end_str, val_split=None):

    print("before time_deltas_from_log")
    dev_ids_begin, deltas = time_deltas_from_log(log_file, begin_str, end_str)
    print("after time_deltas_from_log")

    creation_times = []
    creation_times_per_day = []
    total_exs = []
    print("input_metas: {}".format(input_metas))
    print("dev_ids begin: {}".format(dev_ids_begin))
    for d, dev_id in enumerate(dev_ids_begin):
        logger.info("dev_id %s", dev_id)
        print("devid {}".format(dev_id))
        v = input_metas[dev_id]
        print("v {}".format(v))
        ex_per_day = v["ex_per_day"]
        ex_days = len(v["ex_dates"])
        total_ex = round(ex_days * ex_per_day * 1.0 / (1 - val_split))
        total_exs.append(total_ex)

        ex_creation_time = deltas[d]
        creation_times.append(ex_creation_time)
        creation_times_per_day.append(ex_creation_time / ex_days)
        assert len(creation_times) > 0
        assert len(creation_times_per_day) > 0
        assert len(total_exs) > 0

    ex_creation_profiled = {
        "creation_time_per_day": creation_times_per_day,
        "ex_creation_time": creation_times,
        "total_exs": total_exs,
        "dev_id": dev_ids_begin,
    }
    profile_df = pd.DataFrame.from_dict(ex_creation_profiled)
    profile_df.set_index("dev_id", inplace=True)
    print("len of df: {}".format(len(profile_df)))
    return profile_df


def profile_training_local(log_file, input_metas, begin_str, end_str, val_split=None):

    dev_ids_begin, deltas = time_deltas_from_log(log_file, begin_str, end_str)

    training_times = []
    training_times_per_day = []
    training_times_per_ex = []
    total_exs = []
    days = []
    for d, dev_id in enumerate(dev_ids_begin):
        logger.info("dev_id %s", dev_id)
        v = input_metas[dev_id]
        ex_per_day = v["ex_per_day"]
        ex_days = len(v["ex_dates"])
        total_ex = round(ex_days * ex_per_day * 1.0 / (1 - val_split))
        total_exs.append(total_ex)

        training_time = deltas[d]
        training_times.append(training_time)
        training_times_per_day.append(training_time / ex_days)
        training_times_per_ex.append(training_time / total_ex)
        days.append(ex_days)
    train_profiled = {
        "training_time_per_day": training_times_per_day,
        "training_time_per_ex": training_times_per_ex,
        "training_time": training_times,
        "total_exs": total_exs,
        "total_days": days,
        "dev_id": dev_ids_begin,
    }
    profile_df = pd.DataFrame.from_dict(train_profiled)
    profile_df.set_index("dev_id", inplace=True)
    return profile_df


def profile_training_paperspace(
    jobid,
    headers,
    train_func="LSTM_model",
    log_desc="lstm",
    quilt_user=None,
    datapkg=None,
):
    machine_stats = job_machine_stats(jobid, headers)

    dev_ids, deltas = training_times_from_logs(
        jobid, headers, train_func=train_func, log_desc=log_desc
    )

    package = "/".join([quilt_user, datapkg])
    pkg = quilt.load(package)
    metadata = pkg["metadata"]

    stop_epoch_dict, max_epochs = training_epochs_from_logs(jobid, headers)

    training_times = []
    training_times_per_day = []
    training_times_per_ex = []
    training_times_per_epoch = []
    training_exs = []
    days = []
    stop_epochs = []

    for dev, training_time in zip(dev_ids, deltas):
        training_times
        training_times.append(training_time)
        meta_node = metadata["n" + str(dev)]

        ex_days = np.size(meta_node["ex_dates"]())
        training_times_per_day.append(training_time / ex_days)
        days.append(ex_days)

        train_exs = meta_node["num_train"]().item()
        training_exs.append(train_exs)
        training_times_per_ex.append(training_time / train_exs)

        stop_epoch = stop_epoch_dict[dev]
        training_times_per_epoch.append(training_time / stop_epoch)
        stop_epochs.append(stop_epoch)

    train_profiled = {
        "training_time_per_day": training_times_per_day,
        "training_time_per_ex": training_times_per_ex,
        "training_time_per_epoch": training_times_per_epoch,
        "training_time": training_times,
        "total_exs": training_exs,
        "total_days": days,
        "stop_epoch": stop_epochs,
        "max_epochs": [max_epochs] * len(dev_ids),
        "dev_id": dev_ids,
    }
    profile_df = pd.DataFrame.from_dict(train_profiled)
    profile_df.set_index("dev_id", inplace=True)
    return profile_df


def job_machine_stats(jobid, headers, project="tlproject"):
    payload = {"project": project}
    list_job = "/".join([API_DOMAIN, "jobs", "getJobs"])
    r = requests.get(list_job, params=payload, headers=headers)
    job_stats = r.json()

    for job in job_stats:
        if job["id"] == jobid:
            exit_code = job["exitCode"]
            container = job["container"]
            machine_type = job["machineType"]
            usage = job["usageRate"]
            cpu = job["cpuModel"]
            cpu_count = job["cpuCount"]
            cpu_mem = job["cpuMem"]

    stats = Jobstats(
        jobid=jobid,
        machine_type=machine_type,
        exit_code=exit_code,
        container=container,
        usage_rate=usage,
        cpu_count=cpu_count,
        cpu_mem=cpu_mem,
        cpu_model=cpu,
    )

    return stats


def training_times_from_logs(jobid, headers, train_func="LSTM_model", log_desc="lstm"):
    payload = {"jobId": jobid}
    log_call = "/".join([LOG_DOMAIN, "jobs", "logs"])
    r = requests.get(log_call, params=payload, headers=headers)
    log_content = r.json()

    begin_str = " ".join(
        [train_func + "()", "begin", log_desc]
    )  # 'LSTM_model() begin lstm'
    end_str = " ".join([train_func + "()", "end", log_desc])  # 'LSTM_model() end lstm'

    log_times_msgs = [(pd.Timestamp(l["timestamp"]), l["message"]) for l in log_content]

    lstm_begins = [t_msg[0] for t_msg in log_times_msgs if begin_str in t_msg[1]]
    lstm_ends = [t_msg[0] for t_msg in log_times_msgs if end_str in t_msg[1]]
    dev_ids = [
        logged[1].split()[-1] for logged in log_times_msgs if begin_str in logged[1]
    ]
    training_times = [e - b for b, e in zip(lstm_begins, lstm_ends)]
    return dev_ids, training_times


def training_epochs_from_logs(
    jobid, headers, train_func="summarize_device_training", stop_desc="stopped_epoch"
):
    payload = {"jobId": jobid}
    log_call = "/".join([LOG_DOMAIN, "jobs", "logs"])
    r = requests.get(log_call, params=payload, headers=headers)
    log_content = r.json()

    epochs_str = " ".join([train_func + "()", "{'" + stop_desc])

    log_times_msgs = [l["message"] for l in log_content if epochs_str in l["message"]]
    epochs_msgs = (
        msg[msg.find("{") : msg.find("}") + 1]
        for msg in log_times_msgs
        if msg.find("{") >= 0
    )

    info_dicts = (ast.literal_eval(msg) for msg in epochs_msgs)
    ids_stops = dict([(info["id"], info["stopped_epoch"] + 1) for info in info_dicts])
    max_epoch = None
    for msg in log_times_msgs:
        if "max_epochs" in msg:
            max_epoch = int(msg[msg.find("max_epochs") + 12 : msg.find(", 'id")])
            break

    return ids_stops, max_epoch


def time_deltas_from_log(log_file, begin_str, end_str):
    begins = []
    ends = []
    dev_ids_begin = []
    dev_ids_end = []
    prev_id_begin = None
    prev_id_end = None
    print("begin_str: {}".format(begin_str))

    with open(log_file) as f:
        for logged in f:
            print("logged: {}".format(logged))
            dev_id = logged.split()[-1]
            print("dev_id: {}".format(dev_id))
            begin_str_in_logged = begin_str in logged
            dev_id_in_logged = dev_id in logged
            print("begin_str in logged: {}".format(begin_str_in_logged))
            print("dev_id in logged: {}".format(dev_id_in_logged))
            if all((begin_str in logged, dev_id in logged, dev_id != prev_id_begin)):
                raw_datetime = logged.split()[0:2]
                begins.append(raw_datetime)

                dev_ids_begin.append(dev_id)
                prev_id_begin = dev_id

            elif all((end_str in logged, dev_id in logged, dev_id != prev_id_end)):
                end_str_in_logged = end_str in logged
                print("end_str in logged: {}".format(end_str_in_logged))
                raw_datetime = logged.split()[0:2]
                ends.append(raw_datetime)

                dev_ids_end.append(dev_id)
                prev_id_end = dev_id

        assert dev_ids_begin == dev_ids_end

        deltas = []

        for b, e in zip(begins, ends):
            begin_t, end_t = (pd.Timestamp(" ".join(t)) for t in (b, e))
            assert end_t > begin_t
            deltas.append(end_t - begin_t)

        print("dev_ids_begin: {}".format(dev_ids_begin))
        print("deltas: {}".format(deltas))

        return dev_ids_begin, deltas


if __name__ == "__main__":
    pps_conf = config.get("paperspace")
    script_runner = pps_conf[
        "script_runner"
    ]  # 'gradient' (formerly 'paperspace-python')
    command = pps_conf["command"]  # For example: 'experiments run singlenode'
    ps_command_1, ps_command_2, ps_command_3 = tuple(command.split(" "))
    # Experiment name is job label in Gradient Jobs Console  (JOB ID is automatic)
    experiment_name = pps_conf["experiment"]
    # Is in URL after clicking project name in Gradient Jobs Console
    project_id = pps_conf["project_id"]
    ignore_files = pps_conf["ignorefiles"]
    script = pps_conf["script"]
    zone = pps_conf["zone"]
    machine_type = pps_conf["machinetype"]

    docker_conf = config.get("docker")
    docker_url = docker_conf["docker_url"]  # Such as 'docker.io'
    image = docker_conf["image"]
    tag = docker_conf["tag"]

    data_conf = config.get("data")
    data_opt = data_conf["data_file"]

    model_conf = config.get("model")
    model_opt = model_conf["model_type"]

    opts_str = " ".join([data_opt, model_opt])

    train_groups = config.get("data_entities")
    # Dict in the form of a string that can be eval'd,
    # with ints as keys, and strings containing
    # comma-separated ints (representing IDs) as values (e.g., '1,2')
    train_groups = eval(train_groups["ids_to_train"])

    run_parallel_training(
        script_runner,
        ps_command_1,
        ps_command_2,
        ps_command_3,
        experiment_name,
        script,
        train_groups,
        opts_str,
        project_id,
        zone,
        machine_type,
        docker_url,
        image,
        tag,
        workspace=".",
        ignore_files=ignore_files,
        num_jobs=0,
    )
