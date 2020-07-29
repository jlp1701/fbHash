import subprocess
import re
import os
import json
import random
import numpy as np
import math
from fbHash import fbHashB
import ssdeep
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import HashWrapper as hw
from sklearn.model_selection import KFold
from multiprocessing import Pool



# function to generate fragment (sequential or random) from file
def get_fragments(data, sizes, random_pos=False):
    frag = []
    random.seed()
    for s in sizes:
        length = math.ceil(len(data) * s / 100.0)
        d = 0
        if random_pos:
            d = random.randint(0, len(data) - length)
        frag.append(data[d:(d + length - 1)])
    return frag


def get_fragment_indices(f_path, sizes, random_pos=False):
    indices = []
    random.seed()
    for s in sizes:
        f_len = os.path.getsize(f_path)
        length = math.ceil(f_len * s / 100.0)
        d = 0
        if random_pos:
            d = random.randint(0, f_len - length)
        indices.append((d, d + length - 1))
    return indices


def gen_kfold_files(files, n_folds):
    kf = KFold(n_folds)
    for train, test in kf.split(files):
        yield ([files[i] for i in train], [files[i] for i in test])


##############################################################################################################

w_path = "./weights_1000_all.db"
def fbHashB_hashd(data):
    return fbHashB.hashd_weight_file(data, w_path)
sdhash = hw.HashWrapper("sdhash", [""], ["-t", "-1", "-c"], r".*?\|.*?\|(\d{3})")
schemes = [('fbHashB', fbHashB_hashd, fbHashB.compare), ('ssdeep', ssdeep.hash, ssdeep.compare), ('sdhash', sdhash.hashd, sdhash.compare)]
files_path = "./tests/files/t5-corpus/t5/*.text"


def mrshv2_hc(d1, d2):
    # create temporary files
    path_bin = "/home/burndi/mrsh_v2.0/mrsh_v2.0/mrsh"
    args_hash_and_compare = ["-f", "-t", "0", "-c"]
    compare_regex = r".*? \| .*? \| (\d{3}) "
    file_path = [f"/dev/shm/hashd_{str(random.randint(0, 1000000))}.txt" for _ in range(2)]
    with open(file_path[0], "wb") as f:
        f.write(d1)
    with open(file_path[1], "wb") as f:
        f.write(d2)
    try:
        cmd = [f"{path_bin}"]
        cmd.extend(args_hash_and_compare)
        cmd.append(file_path[0])
        cmd.append(file_path[1])
        # print(f"cmd: {cmd}")
        proc_ret = subprocess.run(cmd, stdout=subprocess.PIPE)
        # print(f"stdout: {proc_ret.stdout}")
        if proc_ret.returncode != 0:
            raise Exception(f"Hash and compare program returned error code: {proc_ret.returncode}")
        if proc_ret.stdout == b'':
            raise Exception(f"No output for input files: {file_path}")
    finally:
        os.remove(file_path[0])
        os.remove(file_path[1])
    m = re.match(compare_regex, str(proc_ret.stdout))
    if m is None:
        raise Exception(f"Output couldn't be parsed: '{proc_ret.stdout}'")

    return float(m.group(1))


def fd_kfold(frag_sizes):
    num_folds = 5
    res = []
    # files_path = "./tests/files/t5-corpus/t5/*.text"
    files_path = "/dev/shm/t5-corpus/t5/*.text"
    min_size = 512 * 100  # for sdhash
    files = list(filter(lambda f: os.path.getsize(f) > min_size, glob.glob(files_path)))
    random.shuffle(files)
    print(f"num file pool: {len(files)}")
    # reduce number of files for testing
    # files = files[0:20]

    kf = gen_kfold_files(files, num_folds)
    for train, test in kf:
        # compute doc weights
        print("generate doc weights")
        docw = fbHashB.compute_document_weights(train)

        def fbHashB_hashd(data):
            return fbHashB.hashd_weights(data, docw)
        # sdhash = hw.HashWrapper("sdhash", [""], ["-t", "-1", "-c"], r".*?\|.*?\|(\d{3})")
        schemes = [('fbHashB', fbHashB_hashd, fbHashB.compare, None), ('ssdeep', ssdeep.hash, ssdeep.compare, None), ('mrshv2', None, None, mrshv2_hc)]

        # compute fragment detection
        print("compute fragment detection")
        r = fragment_detection(schemes, test, frag_sizes)
        res.append(r)
        print(f"res:\n{res}")

    # compose results
    result = np.array(res)
    result = np.average(result, axis=0)
    return result


def fragment_detection(schemes, files, frag_sizes):
    num_files = min(len(files), 20)
    print(f"number files: {num_files}")
    # frag_sizes = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 4, 3, 2, 1]
    # frag_sizes = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 4, 3, 2, 1]
    # frag_sizes = [95, 75, 55, 50, 35, 15]
    random_pos = True

    # pick files and fragments
    picked_fi_and_fr = fdet_pick_files_and_fragments(files, num_files, frag_sizes, random_pos)
    #print(f"picked_fi_and_fr: {picked_fi_and_fr}")

    # read in the data

    shape = (0, 2, len(frag_sizes))
    res = np.zeros(shape, dtype=float)
    for (n, h, c, hc) in schemes:
        print(f"scheme: {n}")
        # compute similarity scores
        if h and c:
            frag_data = fdet_read_and_hash_data(picked_fi_and_fr, h)
            scores = fdet_compute(frag_data, c)
        else:
            scores = fdet_read_hash_compute(picked_fi_and_fr, hc)
        #with np.printoptions(suppress=True, precision=1):
        #    print(f"scores: {scores}")

        # calc match percentage and avg similarity
        m_perc = fdet_get_match_perc(scores)
        #print(f"m_perc: {m_perc}")

        avg_sim = fdet_get_avg_similarity(scores)
        #with np.printoptions(suppress=True, precision=1):
        #    print(f"avg_sim: {avg_sim}")

        # res.append({'scheme': n, 'match_perc': m_perc, 'avg_score': avg_sim})
        res = np.append(res, [[m_perc, avg_sim]], axis=0)
        # print(f"res:\n{res}")
    return res


def fdet_pick_files_and_fragments(input_files, num_files, frag_sizes, random_pos):
    """Randomly picks files from the directory and choses fragments according to the
        parameters. Returns a structure with picked file info:
        [{'file_path': fpath, 'frag_pos': [(i_beg, i_end), (i_beg, i_end), ...]},
         {'file_path': fpath, 'frag_pos': [(i_beg, i_end), (i_beg, i_end), ...]}, ...]
    """
    files = random.sample(input_files, num_files)

    # select fragment indices
    picked_data = []
    for f in files:
        file_data = {'file_path': f, 'frag_pos': get_fragment_indices(f, frag_sizes, random_pos)}
        picked_data.append(file_data)
    return picked_data


def fdet_read_and_hash_data(fdet_param_struct, hashd):
    """Reads the data according to the input info structure generated by
        'fdet_pick_files_and_fragments()'
        Returns read in data in structure:
        [{'file': data, 'fragments': [data_f1, data_f2, ...]}, 
         {'file': data, 'fragments': [data_f1, data_f2, ...]}, ...]
    """
    fragment_data = []
    for f_param in fdet_param_struct:
        # read in whole file:
        with open(f_param['file_path'], "rb") as f:
            fdata = f.read()
        # slice fragment data
        fragments = []
        for (s, e) in f_param['frag_pos']:
            fragments.append(hashd(fdata[s:e]))
        fragment_data.append({'file': hashd(fdata), 'fragments': fragments})
    return fragment_data


def fdet_compute(fdet_data_struct, compare):
    """Computes the comparisons and calculates the similarites according
        to the fragment detection algorithm.
        Input: fragment data structure generated by fdet_read_data, hashd = anonymous hash function
        Output: n x n x k numpy array (n = number of files, k = number of fragments for each file)
    """
    n = len(fdet_data_struct)
    k = len(fdet_data_struct[0]['fragments'])

    scores = np.zeros((n, n, k), dtype=float)
    for n_row in range(n):
        for n_col in range(n):
            for k_col in range(k):
                scores[n_row, n_col, k_col] = compare(fdet_data_struct[n_row]['file'], fdet_data_struct[n_col]['fragments'][k_col])
    return scores



def fdet_read_hash_compute(fdet_param_struct, hc):
    fragment_data = []
    for f_param in fdet_param_struct:
        # read in whole file:
        with open(f_param['file_path'], "rb") as f:
            fdata = f.read()
        # slice fragment data
        fragments = []
        for (s, e) in f_param['frag_pos']:
            fragments.append(fdata[s:e])
        fragment_data.append({'file': fdata, 'fragments': fragments})

    fdet_data_struct = fragment_data

    n = len(fdet_data_struct)
    k = len(fdet_data_struct[0]['fragments'])

    scores = np.zeros((n, n, k), dtype=float)
    for n_row in range(n):
        for n_col in range(n):
            for k_col in range(k):

                scores[n_row, n_col, k_col] = hc(fdet_data_struct[n_row]['file'], fdet_data_struct[n_col]['fragments'][k_col])
    return scores

def fdet_get_match_perc(fdet_comp_array):
    """Gets the comparison matrix of a fragment detection run and calculates the
        match percentage for each fragment size
        Input: fragment comparison array generated by fdet_compute
        Output: list with match percentage for each fragment size
    """
    # for each fragment size:
    num_files = fdet_comp_array.shape[0]
    num_frag_sizes = fdet_comp_array.shape[2]
    m_perc = []
    for i in range(num_frag_sizes):
        # get all values of fragment size
        f_res = fdet_comp_array[:, :, i].copy()  # result is a n x n matrix
        diag = f_res.diagonal().copy()
        np.fill_diagonal(f_res, 0)
        num_gen = (diag > f_res.max(axis=1)).sum()  # sum up the occurences of zeros (= correctly matched)
        m = num_gen / num_files * 100
        m_perc.append(m)
    return np.array(m_perc, dtype=float)


def fdet_get_avg_similarity(fdet_comp_array):
    """Gets the comparison matrix of a fragment detection run and calculates the 
        average similarity for each fragment size
        Input: fragment comparison array generated by fdet_compute
        Output: list with average similarity score for each fragment size
    """
    num_files = fdet_comp_array.shape[0]
    num_frag_sizes = fdet_comp_array.shape[2]
    avg_sim = np.zeros(num_frag_sizes)
    for i in range(num_files):
        avg_sim += fdet_comp_array[i, i]
    avg_sim = avg_sim / num_files
    return avg_sim


def print_results(title, frag_sizes, schemes, results):
    """Prints the chart for fragment detection
        Input:  frag_sizes: list of fragment sizes used
                results: [{'scheme': name, 'match_perc': [...], 'avg_score': [...]},
                          {'scheme': name, 'match_perc': [...], 'avg_score': [...]}, ...]
    """
    if len(schemes) > 3:
        raise Exception("Not enough colors.")

    width = 0.2
    c = ['g', 'b', 'r']
    o = [- width, 0, width]

    x = np.arange(len(frag_sizes))
    fig, ax = plt.subplots()

    for i in range(len(schemes)):
        ax.plot(x, results[i][0], '--', c=c[i])
        ax.bar(x + o[i], results[i][1], width * 4 / 5, color=c[i], label=schemes[i])
    ax.set_ylim([0, 101])
    ax.set_xticks(x)
    ax.set_xticklabels(list(map(round, frag_sizes)))
    ax.legend()
    ax.set(xlabel='fragment sizes [%]', ylabel='avg score', title=title)
    ax.grid(axis='y')

    # fig.savefig("test.png")
    plt.show()

########################################################################################################


def cbd_kfold(frag_sizes):
    num_folds = 5
    res = []
    # files_path = "./tests/files/t5-corpus/t5/*.text"
    files_path = "/dev/shm/t5-corpus/t5/*.text"
    min_size = 512 * 100  # for sdhash
    files = list(filter(lambda f: os.path.getsize(f) > min_size, glob.glob(files_path)))
    random.shuffle(files)
    # reduce number of files for testing
    # files = files[0:30]

    kf = gen_kfold_files(files, num_folds)
    for train, test in kf:
        # compute doc weights
        print("generate doc weights")
        docw = fbHashB.compute_document_weights(train)

        def fbHashB_hashd(data):
            return fbHashB.hashd_weights(data, docw)
        sdhash = hw.HashWrapper("sdhash", [""], ["-t", "-1", "-c"], r".*?\|.*?\|(\d{3})")
        schemes = [('fbHashB', fbHashB_hashd, fbHashB.compare), ('ssdeep', ssdeep.hash, ssdeep.compare), ('sdhash', sdhash.hashd, sdhash.compare)]

        # compute fragment detection
        print("compute common block detection")
        r = common_block_detection(schemes, test, frag_sizes)
        res.append(r)
        print(f"res:\n{res}")

    # compose results
    result = np.array(res)
    result = np.average(result, axis=0)
    return result


def common_block_detection(schemes, files, frag_sizes):
    runs = min(len(files), 30)

    data = cbd_pick_files_and_fragments_random(files, frag_sizes, runs)
    print(f"files picked")

    shape = (0, 2, len(frag_sizes))
    res = np.zeros(shape, dtype=float)
    for (n, h, c) in schemes:
        scores = cbd_compute(data, h, c)
        print(f"scores:\n{scores}")

        m_perc = cbd_get_match_perc(scores)
        avg = cbd_get_avg_similarity(scores)

        # with np.printoptions(suppress=True, precision=1):
        print(f"m_perc:\n{m_perc}")
        print(f"avg_sim:\n{avg}")

        res = np.append(res, [[m_perc, avg]], axis=0)
    return res


def cbd_pick_files_and_fragments(file_set, frag_sizes, runs):
    """Generate a structure containing the files with injected fragments to be compared
        Return: [[(f1, f2), (f1, f2), ...],
                 [(f1, f2), (f1, f2), ...], ... ]
    """
    file_set_sorted = [(f, os.path.getsize(f)) for f in file_set]
    # print(f"file_set_sorted: {file_set_sorted}")
    res = []
    r = 0
    while r < runs:
        # pick src file randomly
        f_src = random.choice(file_set_sorted)
        # print(f"f_src: {f_src}")
        # filter the set of files according to the size of src
        sink_set = [f for f in file_set_sorted if f_src[1] * 0.9 <= f[1] <= f_src[1] * 1.1 and f != f_src]  # select files within a 10% file size margin
        # print(f"files in sink_set: {len(sink_set)}")
        if len(sink_set) < 2:
            print(f"Not enouth files available as sink!")
            continue
        f_sinks = random.sample(sink_set, 2)

        print(f"picked: src: {f_src[1]}")
        print(f"picked: sink: {f_sinks[0][1]}, {f_sinks[1][1]}")
        # select random fragments
        fragments = get_fragment_indices(f_src[0], frag_sizes, random_pos=True)
        with open(f_src[0], "rb") as f:
            fdata = f.read()
        data_sinks = []
        with open(f_sinks[0][0], 'rb') as f:
            data_sinks.append(f.read())
        with open(f_sinks[1][0], 'rb') as f:
            data_sinks.append(f.read())

        run = []
        p0 = random.randint(0, f_sinks[0][1])
        p1 = random.randint(0, f_sinks[1][1])
        for (s, e) in fragments:
            run.append((data_sinks[0][0:p0] + fdata[s:e] + data_sinks[0][p0:], data_sinks[1][0:p1] + fdata[s:e] + data_sinks[1][p1:]))
        res.append(run)
        # remove src and sink files from set
        file_set_sorted.remove(f_src)
        file_set_sorted.remove(f_sinks[0])
        file_set_sorted.remove(f_sinks[1])
        r += 1
    return res


def cbd_pick_files_and_fragments_random(file_set, frag_sizes, runs):
    """Generate a structure random data with injected fragments to be compared
        Return: [[(f1, f2), (f1, f2), ...],
                 [(f1, f2), (f1, f2), ...], ... ]
    """
    file_set_sorted = [(f, os.path.getsize(f)) for f in file_set]
    # print(f"file_set_sorted: {file_set_sorted}")
    res = []
    for _ in range(runs):
        # pick src file randomly
        f_src = random.choice(file_set_sorted)
        file_set_sorted.remove(f_src)

        # select random fragments
        fragments = get_fragment_indices(f_src[0], frag_sizes, random_pos=True)
        with open(f_src[0], "rb") as f:
            fdata = f.read()

        # generate random data for sinks
        data_sinks = [os.getrandom(f_src[1]) for _ in range(2)]

        run = []
        p0 = random.randint(0, len(data_sinks[0]))
        p1 = random.randint(0, len(data_sinks[1]))
        for (s, e) in fragments:
            run.append((data_sinks[0][0:p0] + fdata[s:e] + data_sinks[0][p0:], data_sinks[1][0:p1] + fdata[s:e] + data_sinks[1][p1:]))
        res.append(run)
    return res



def cbd_compute(file_pair_struct, hashd, compare):
    """Hash and compute similarity for each pair sharing a common block
        Return: 
    """
    runs = len(file_pair_struct)
    sizes = len(file_pair_struct[0])
    scores = np.zeros((sizes, runs, runs), dtype=float)
    for s in range(sizes):
        h = []
        for r in range(runs):
            h.append((hashd(file_pair_struct[r][s][0]), hashd(file_pair_struct[r][s][1])))

        for r_row in range(runs):
            for r_col in range(runs):
                # print(f"r_col: {r_col}")
                scores[s, r_row, r_col] = compare(h[r_row][0], h[r_col][1])
    return scores


def cbd_get_match_perc(scores):
    """Compute the match percentage for each file size
        Return: [s0_match_perc, ... , sn_match_perc]
    """
    # for each fragment size:
    num_frag_sizes = scores.shape[0]
    num_runs = scores.shape[1]
    m_perc = []
    for i in range(num_frag_sizes):
        # get all values of fragment size
        f_res = scores[i].copy()
        diag = f_res.diagonal().copy()
        np.fill_diagonal(f_res, 0)
        num_gen = (diag > f_res.max(axis=1)).sum()  # sum up the occurences of zeros (= correctly matched)
        m = num_gen / num_runs * 100
        m_perc.append(m)
    return m_perc


def cbd_get_avg_similarity(scores):
    """
    """
    sizes = scores.shape[0]
    runs = scores.shape[1]
    avg = []
    for s in range(sizes):
        avg.append(scores[s].diagonal().sum() / runs)
    return avg


def cbd_print_results(title, frag_sizes, results):
    """Prints the chart for common block detection
        Input:  frag_sizes: list of fragment sizes used
                results: [{'scheme': name, 'match_perc': [...], 'avg_score': [...]},
                          {'scheme': name, 'match_perc': [...], 'avg_score': [...]}, ...]
    """
    if len(results) > 3:
        raise Exception("Not enough colors.")

    width = 0.2
    c = ['g', 'b', 'r']
    o = [- width, 0, width]

    frag_sizes = list(map(lambda t: (t / (t + 100.0) * 100), frag_sizes))
    x = np.arange(len(frag_sizes))
    fig, ax = plt.subplots()

    for i in range(len(results)):
        ax.plot(x, results[i]['match_perc'], '--', c=c[i])
        ax.bar(x + o[i], results[i]['avg_score'], width * 4 / 5, color=c[i], label=results[i]['scheme'])
    ax.set_ylim([0, 101])
    ax.set_xticks(x)
    ax.set_xticklabels(list(map(round, frag_sizes)))
    ax.legend()
    ax.set(xlabel='fragment sizes [%]', ylabel='avg score', title='Single-common Block Detection')
    ax.grid(axis='y')

    # fig.savefig("test.png")
    plt.show()


def main():
    schemes_names = ["fbHashB", "ssdeep", "sdhash"]
    frag_sizes = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 4, 3, 2, 1]
    fd_res = fd_kfold(frag_sizes)
    print(f"fd_res: {fd_res}")
    s = {}
    s['schemes'] = schemes_names
    s['frag_sizes'] = frag_sizes
    s['fd_results'] = fd_res.tolist()
    with open("fd_res.json", "w") as f:
        json.dump(s, f)
    # print_results("Fragment Detection (5-fold CV)", frag_sizes, schemes_names, fd_res)
    

    schemes_names = ["fbHashB", "ssdeep", "sdhash"]
    frag_sizes = [100, 66.66, 42.86, 25, 11.11, 5.2, 4.1, 3.09, 2.04, 1.01]
    cbd_res = cbd_kfold(frag_sizes)
    print(f"cbd_res: {cbd_res}")
    s = {}
    s['schemes'] = schemes_names
    s['frag_sizes'] = frag_sizes
    s['cbd_results'] = cbd_res.tolist()
    with open("cbd_res.json", "w") as f:
        json.dump(s, f)
    # print_results("Single-common block detection (5-fold CV)", frag_sizes, schemes_names, cbd_res)


if __name__ == '__main__':
    main()
