import pyds9
import subprocess
import time
import os


def start_ds9():
    """
    Start DS9
    """
    # os.environ["XPA_PORT"] = "DS9:ds9 12345 12346"
    os.environ["SAS_VERBOSITY"] = "5"

    targets = pyds9.ds9_targets()
    if targets is None:
        print("Starting DS9")
        command = subprocess.Popen(
            ["ds9"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        time.sleep(25)
        targets = pyds9.ds9_targets()
    else:
        print("DS9 is already running")
    print(targets)
    address = targets[0][8:]
    python_ds9 = pyds9.DS9(address)
    return python_ds9
