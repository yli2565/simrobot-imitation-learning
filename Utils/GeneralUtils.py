import psutil


def is_zombie(pid):
    try:
        proc = psutil.Process(pid)
        return proc.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False  # Process does not exist


def kill_process(pid):
    try:
        p = psutil.Process(pid)
        p.terminate()  # Politely ask the process to terminate
        p.wait(timeout=3)  # Wait up to 3 seconds for the process to end
    except psutil.NoSuchProcess:
        print(f"No process with PID {pid} exists.")
    except psutil.TimeoutExpired:
        print(f"Process {pid} did not terminate in time; trying to kill it.")
        p.kill()  # Forcefully kill the process
        p.wait()  # Wait for the process to be killed
        print(f"Process {pid} has been forcefully killed.")
    except Exception as e:
        print(f"Error killing process {pid}: {str(e)}")
