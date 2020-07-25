import os
import subprocess
import random
import re


class HashWrapper(object):
    """docstring for HashWrapper"""

    def __init__(self, path_bin, args_hash, args_compare, compare_regex):
        super(HashWrapper, self).__init__()
        self.path_bin = path_bin
        self.args_hash = args_hash
        self.args_compare = args_compare
        self.compare_regex = compare_regex

    def hashf(self, file_path):
        cmd = [f"{self.path_bin}"]
        cmd.extend(self.args_hash)
        cmd.append(f"{file_path}")
        proc_ret = subprocess.run(cmd, stdout=subprocess.PIPE)
        # print(f"stdout: {proc_ret.stdout}")
        if proc_ret.returncode != 0:
            raise Exception(f"Hash program returned error code: {proc_ret.returncode}")
        if proc_ret.stdout == b'':
            raise Exception(f"No output for input file: {file_path}")
        return proc_ret.stdout

    def hashd(self, data):
        # create temporary file
        suff = str(random.randint(0, 1000000))
        file_path = f"/tmp/hashd_{suff}.txt"
        with open(file_path, "wb") as f:
            f.write(data)
        try:
            h = self.hashf(file_path)
        finally:
            os.remove(file_path)
        return h

    def compare(self, h1, h2):
        # write both hashes to temp file
        suff = str(random.randint(0, 1000000))
        file_path = f"/tmp/compare_{suff}.txt"
        with open(file_path, "wb") as f:
            f.write(h1)
            f.write(h2)
        # compare
        # cmd = [f"{self.path_bin}", "-t", "0", "-c", f"{file_path}"]
        cmd = [self.path_bin]
        cmd.extend(self.args_compare)
        cmd.append(file_path)
        # print(f"cmd: {cmd}")
        proc_ret = subprocess.run(cmd, stdout=subprocess.PIPE)
        os.remove(file_path)
        # print(f"stdout: {proc_ret.stdout}")
        if proc_ret.returncode != 0:
            raise Exception(f"Compare program returned error code: {proc_ret.returncode}")
        if proc_ret.stdout == b'':
            raise Exception(f"No output for input file: {file_path}")
        m = re.match(self.compare_regex, str(proc_ret.stdout))
        if m is None:
            raise Exception(f"Output couldn't be parsed.")

        return int(m.group(1))


# if __name__ == '__main__':
#     hw = HashWrapper("sdhash", [""], ["-t", "0", "-c"], r".*?\|.*?\|(\d{3})")
#     for i in range(1):
#         print(i)
#         h1 = hw.hashf("a.txt")
#         h2 = hw.hashf("b.txt")
#         # print(f"h1: {h1}")
#         # print(f"h2: {h2}")
#         print(f"comparison: {hw.compare(h1, h2)}")
