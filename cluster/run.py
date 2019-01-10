#!/usr/bin/env python3

#-------------------------------------------------------------------------
# run
#
# This file simplifies the process of sending jobs to the cluster.
# It parses input arguments that describe how the jobs should be
# submitted, writes a bash script to a file, and finally calls qsub
# with that bash script as an argument.
#
# When qsub runs the script, the first thing it does is source a
# virtualenv script that configures the python environment properly.
#-------------------------------------------------------------------------

import argparse
import datetime
import os
import re
import subprocess
import sys
import time

defaultjob = 'run'

def parse_args():
    # Parse input arguments
    #   Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--command', help='The command to run (e.g. "python -m module.name --arg=value")', type=str, required=True)
    parser.add_argument('--jobname', help='A name for the job (max 10 chars)', type=str, default=defaultjob)
    parser.add_argument('--jobtype', help='Which type of job to request', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--env', help='Path to virtualenv', type=str, default='./env')
    parser.add_argument('--nresources', help='Number of CPUs / GPUs to request', type=int, default=1)
    parser.add_argument('--duration', help='Expected duration of job', choices=['test', 'short', 'long', 'vlong'], default='vlong')
    parser.add_argument('-t','--taskid', help='Task ID of first task', type=int, default=1)
    parser.add_argument('-n','--ntasks', help='Number of tasks', type=int, default=0)
    parser.add_argument('-max','--maxtasks', help='Maximum number of simultaneous tasks', type=int, default=-1)
    parser.add_argument('-y','--dry_run', help="Don't actually submit jobs to grid engine", action='store_true')
    parser.set_defaults(dry_run=False)
    parser.add_argument('--email', help='Email address(es) to notify when job is complete: addr1@brown.edu[, addr2@brown.edu]', type=str, default=None)
    parser.add_argument('--hold_jid', help='Hold job until the specified job ID has finished', type=int, default=None)
    return parser.parse_args()
args = parse_args()

def launch():
    # Define the bash script that qsub should run (with values
    # that need to be filled in using the input args).
    venv_path = os.path.join(args.env, 'bin', 'activate')
    script_body='''#!/bin/bash

source {}
{} '''.format(venv_path, args.command)

    # GridEngine doesn't like ranges of tasks that start with zero, so if you
    # submit a job with zero tasks, we ignore the taskid variable and submit a
    # single job with no task id instead of using GridEngine's range feature.
    if args.ntasks > 0:
        script_body += r'$SGE_TASK_ID'
    script_body += '\n'

    # Write the script to a file
    os.makedirs("grid/scripts/", exist_ok=True)
    jobfile = "grid/scripts/{}".format(args.jobname)
    with open(jobfile, 'w') as f:
        f.write(script_body)

    # Call the appropriate qsub command. The default behavior is to use
    # GridEngine's range feature, which starts a batch job with multiple tasks
    # and passes a different taskid to each one. If ntasks is zero, only a
    # single job is submitted with no subtasks.
    cmd = 'qsub '
    cmd += '-cwd ' # run script in current working directory

    # When using the Brown grid:
    #  -l test   (10 min, high priority, limited to one slot per machine)
    #  -l short  (1 hour)
    #  -l long   (1 day)
    #  -l vlong  (infinite duration)
    #  -l gpus=# (infinite duration, on a GPU machine)
    if args.jobtype == 'gpu':
        cmd += '-l gpus={} '.format(args.nresources)# Request a single GPU
    else:
        cmd += '-l {} '.format(args.duration)
        if args.nresources > 1:
            cmd += '-pe smp {} '.format(args.nresources) # Request multiple CPUs

    os.makedirs("./grid/logs/", exist_ok=True)
    cmd += '-o ./grid/logs/ ' # save stdout file to this directory
    cmd += '-e ./grid/logs/ ' # save stderr file to this directory

    # The -terse flag causes qsub to print the jobid to stdout. We read the
    # jobid with subprocess.check_output(), and use it to delay the email job
    # until the entire batch job has completed.
    cmd += '-terse '

    if args.ntasks > 0:
        cmd += "-t {}-{} ".format(args.taskid, args.taskid+args.ntasks-1) # specify task ID range
        if args.maxtasks > 0:
            cmd += "-tc {} ".format(args.maxtasks) # set maximum number of running tasks

    # Prevent GridEngine from running this new job until the specified job ID is finished.
    if args.hold_jid is not None:
        cmd += "-hold_jid {} ".format(args.hold_jid)
    cmd += "{}".format(jobfile)

    print(cmd)

    if not args.dry_run:
        try:
            byte_str = subprocess.check_output(cmd, shell=True)
            jobid = int(byte_str.decode('utf-8').split('.')[0])
            if args.email is not None:
                notify_cmd = 'qsub '
                notify_cmd += '-o /dev/null ' # don't save stdout file
                notify_cmd += '-e /dev/null ' # don't save stderr file
                notify_cmd += '-m b ' # send email when this new job starts
                notify_cmd += '-M "{}" '.format(args.email) # list of email addresses
                notify_cmd += '-hold_jid {} '.format(jobid)
                notify_cmd += '-N ~{} '.format(args.jobname[1:]) # modify the jobname slightly
                notify_cmd += '-b y sleep 0' # the actual job is a NO-OP
                subprocess.call(notify_cmd, shell=True)
        except (subprocess.CalledProcessError, ValueError) as e:
            print(e)
            sys.exit()

if args.jobname == defaultjob:
    args.jobname = "run{}".format(args.taskid)
elif not re.match(r'^(\w|\.)+$', args.jobname):
    # We want to create a script file, so make sure the filename is legit
    print("Invalid job name: {}".format(args.jobname))
    sys.exit()
launch()
