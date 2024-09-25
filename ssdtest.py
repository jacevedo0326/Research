import os
import shutil
import psutil

def check_disk_space(path="/"):
    total, used, free = shutil.disk_usage(path)
    print(f"Total: {total // (2**30)} GB")
    print(f"Used: {used // (2**30)} GB")
    print(f"Free: {free // (2**30)} GB")

def check_open_files_limit():
    soft, hard = psutil.Process().rlimit(psutil.RLIMIT_NOFILE)
    print(f"Open file descriptors limit (soft): {soft}")
    print(f"Open file descriptors limit (hard): {hard}")

def check_current_open_files():
    p = psutil.Process()
    print(f"Currently open file descriptors: {p.num_fds()}")

def main():
    print("Checking disk space:")
    check_disk_space()
    print("\nChecking open files limit:")
    check_open_files_limit()
    print("\nChecking current open files:")
    check_current_open_files()

if __name__ == "__main__":
    main()