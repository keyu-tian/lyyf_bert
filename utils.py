import datetime
import logging
import subprocess
import time


def time_str():
    return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def os_system(cmd):
    t0 = time.time()
    cnt = 0
    while time.time() - t0 < 600:
        print(f'(try={cnt})', end='', flush=True)
        sp = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, bufsize=-1)
        t1 = time.time()
        time.sleep(0.5)
        while time.time() - t1 < 60:
            ret = sp.poll()
            if ret is not None:
                print(f'(ret={ret})', end='', flush=True)
                sp.kill()
                return
            time.sleep(3)
        sp.kill()
        time.sleep(10)
        cnt += 1
