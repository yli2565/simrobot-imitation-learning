"""
Common utility functions
"""

import shutil
import psutil
import subprocess
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


def should_use_vglrun():
    def command_exists(cmd):
        return shutil.which(cmd) is not None

    def get_renderer(prefix=""):
        cmd = f"{prefix}glxinfo"
        try:
            result = subprocess.run(
                cmd.split(), capture_output=True, text=True, check=True
            )
            for line in result.stdout.split("\n"):
                if "OpenGL renderer string" in line:
                    return line.split(":", 1)[1].strip()
        except subprocess.CalledProcessError:
            return None
        return None

    def is_software_renderer(renderer):
        return any(sw in renderer.lower() for sw in ["llvmpipe", "softpipe", "swrast"])

    # Check if required commands exist
    if not command_exists("glxinfo") or not command_exists("vglrun"):
        print("Error: glxinfo or vglrun is not installed.")
        return False

    current_renderer = get_renderer()
    vgl_renderer = get_renderer("vglrun ")

    if current_renderer is None or vgl_renderer is None:
        print("Error: Unable to get renderer information.")
        return False

    print(f"Current renderer: {current_renderer}")
    print(f"Renderer with VirtualGL: {vgl_renderer}")

    if is_software_renderer(current_renderer):
        if not is_software_renderer(vgl_renderer):
            print("VirtualGL successfully enabled hardware acceleration!")
            return True
        else:
            print("WARNING: VirtualGL could not enable hardware acceleration.")
            return False
    else:
        if current_renderer != vgl_renderer:
            print(
                "Hardware acceleration is already enabled, but VirtualGL is connecting to a different GPU."
            )
            return True
        else:
            print("Hardware acceleration is already enabled.")
            print("VirtualGL is not necessary in this case, but using it won't hurt.")
            return False
