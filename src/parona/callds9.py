import pyds9
import subprocess
import time
import os

def start_ds9():
    """
    Start DS9
    """
    os.environ["XPA_PORT"] = "DS9:ds9 12345 12346"
    os.environ["SAS_VERBOSITY"] = "5"

    command = subprocess.Popen(
        ['ds9'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(3)
    python_ds9 = pyds9.DS9("7f000001:12345")
    return python_ds9