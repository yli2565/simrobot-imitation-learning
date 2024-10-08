"""
Common utility functions
"""

import psutil
import traceback
import os
import signal


def is_zombie(pid):
    try:
        proc = psutil.Process(pid)
        return proc.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False  # Process does not exist


def kill_process(pid):
    try:
        p = psutil.Process(pid)
        p.terminate()  # Ask the process to terminate
        p.wait(timeout=3)  # Wait up to 3 seconds for the process to end
    except psutil.NoSuchProcess:
        print(f"No process with PID {pid} exists.")
    except psutil.TimeoutExpired:
        print(f"Process {pid} did not terminate in time; trying to kill it.")
        p.kill()  # kill the process
        p.wait()  # Wait for the process to be killed
        print(f"Process {pid} has been forcefully killed.")
    except KeyError:
        print(f"Process {pid} does not exist.")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError as e:
            print(f"Process {pid} is already killed")
    except Exception as e:
        print(f"Error killing process {pid}: {e.__class__.__name__} - {str(e)}")
        traceback.print_exc()
