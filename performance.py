import os
import random
import math
from fbHash import fbHashB
import glob
import matplotlib
import matplotlib.pyplot as plt


# function for selecting random test files (implemented in helper.py)

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


def fragment_detection(dir, w_path, num_files, frag_sizes, random_pos):
    # set up pairings and comparisons to be computed by several tools
    # for all tools:
    # perform hashing of all files and fragments for all tools
    # evaluate comparisons (1) match percentage (2) average similarity score
    frag_data = []
    files = random.sample(glob.glob(dir), num_files)
    print(f"files picked: {files}")

    # read in all files
    for fp in files:
        frag_data.append({'file_path': fp, 'data': fbHashB.hashf(fp, w_path)})

    # compute fragments of all files and hash them
    for fr in frag_data:
        with open(fr['file_path'], "rb") as f:
            data = list(f.read())
        bla = list(map(lambda d: fbHashB.hashd(d, w_path), get_fragments(data, frag_sizes, random_pos)))
        fr['fragments'] = bla

    comp_res = []
    # do comparisons
    for fi in frag_data:
        fj_res = []
        for fj in frag_data:
            frag_res = []
            for frag in fj['fragments']:
                frag_res.append(fbHashB.compare(fi['data'], frag))
            fj_res.append(frag_res)
        comp_res.append(fj_res)

    return {'results': comp_res, 'frag_sizes': frag_sizes}


def analyze_fragment_detection(fragment_detection):
    # calc match percentage and average score of each fragment size
    frag_sizes = fragment_detection['frag_sizes']
    res = fragment_detection['results']

    # print(f"res: {res}")
    # calc match percentage
    # for all fragment sizes
    frag_res = []
    for fsize in range(len(frag_sizes)):
        n_gen_comp = 0
        for i in range(len(res)):
            # get genuine comp result
            gen_comp = res[i][i][fsize]

            # get impostor res and sort them descending
            imp_comp = sorted([c[fsize] for c in res[i] if c != res[i][i]], reverse=True)

            print(f"fsize: {frag_sizes[fsize]}: gen_comp: {gen_comp}; imp_comp: {imp_comp}")
            if gen_comp > imp_comp[0]:
                n_gen_comp += 1
            # print(f"n_gen_comp: {n_gen_comp}; len(res): {len(res)}")
        frag_res.append(n_gen_comp / len(res) * 100.0)

    # calc avg score for all fragments
    avg_score = []
    for i_frag in range(len(frag_sizes)):
        score = []
        for i_f in range(len(res)):
            score.append(res[i_f][i_f][i_frag])  # get the genuine comp scores
        avg_score.append(sum(score) / len(score))

    # plot
    print(f"frag_res: {frag_res}")
    fig, ax = plt.subplots()
    ax.plot(frag_sizes, frag_sizes, c='g')
    ax.plot(frag_sizes, avg_score)
    ax.plot(frag_sizes, frag_res, c='r')

    ax.set(xlabel='fragment sizes', ylabel='avg score',
           title='Fragment detection')
    ax.grid()

    # fig.savefig("test.png")
    plt.show()


def common_block_detection(file_set, frag_sizes, runs, w_path):
    """Generates fragments and does comparisons for one specific frag size (in percent)
    on a set of files.
    """
    # get the length of all files and store them in a list of tuples [(path, size), (path, size),...]
    file_set_sorted = []
    for f in file_set:
        file_set_sorted.append((f, os.path.getsize(f)))
    # sort the list with respect to their size
    file_set_sorted.sort(key=lambda t: t[1])  # sort by filesize
    # print(f"sorted file set: {file_set_sorted}")
    # for number of runs:
    random.seed()
    res = {}
    for i_run in range(runs):
        data_src = []
        data_sinks = [[], []]
        # select three files with similar size
        # pick src file randomly
        f_src = random.sample(file_set_sorted, 1)[0]
        print(f"f_src: {f_src}")
        # filter the set of files
        sink_set = [f for f in file_set_sorted if f_src[1] * 0.9 <= f[1] <= f_src[1] * 1.1 and f != f_src]  # select files within a 10% file size margin
        print(f"files in sink_set: {len(sink_set)}")
        f_sinks = random.sample(sink_set, 2)
        print(f"f_sinks: {f_sinks}")

        with open(f_src[0], 'rb') as f:
            data_src = list(f.read())
        # select two files for fragment sink
        with open(f_sinks[0][0], 'rb') as f:
            data_sinks[0] = list(f.read())
        with open(f_sinks[1][0], 'rb') as f:
            data_sinks[1] = list(f.read())

        # for each fragment size:
        res = {}
        for frag_size in frag_sizes:
            # extract fragment from fragment source
            (data_frag, ) = get_fragments(data_src, [frag_size], random_pos=True)

            # insert fragment data into sinks at random positions
            data_sinks_ins = [list(data_sinks[0]), list(data_sinks[1])]
            rnd_pos = random.randint(0, len(data_sinks[0]))
            data_sinks_ins[0][rnd_pos:rnd_pos] = data_frag
            rnd_pos = random.randint(0, len(data_sinks[1]))
            data_sinks_ins[1][rnd_pos:rnd_pos] = data_frag

            # hash and compare the two sinks
            hash_sink = [fbHashB.hashd(data_sinks_ins[0], w_path), fbHashB.hashd(data_sinks_ins[1], w_path)]
            sim = fbHashB.compare(hash_sink[0], hash_sink[1])

            # normalize the result with respect to the fragment size
            # sim_n =  (frag_size / (frag_size + 100.0)) * 100 / sim
            print(f"frag_size: {frag_size}: sim: {sim}")

            # save result in list
            if frag_size not in res:
                res[frag_size] = [sim]
            else:
                res[frag_size].append(sim)

    # generate average result
    for frag_size in res:
        res[frag_size] = sum(res[frag_size]) / len(res[frag_size])

    return res


def analyze_common_block_detection(results):
    # calc match percentage and average score of each fragment size
    # plot
    frag_sizes = list(map(lambda t: (t / (t + 100.0) * 100), results))
    avg_score = list(map(lambda t: results[t], results))
    print(f"frag_sizes: {frag_sizes}")
    print(f"avg_score: {avg_score}")
    fig, ax = plt.subplots()
    ax.plot(frag_sizes, frag_sizes, c='g')
    ax.plot(frag_sizes, avg_score)
    # ax.plot(frag_sizes, frag_res, c='r')

    ax.set(xlabel='fragment sizes', ylabel='avg score',
           title='Single-common block detection')
    ax.grid()

    # fig.savefig("test.png")
    plt.show()
    pass


def main():
    # with open("./tests/files/t5-corpus/t5/004958.text", "rb") as f:
    #    d = list(f.read())
    # h = {k: v for k, v in sorted(fbHashB.compute_chunk_freq(d).items(), key=lambda item: item[1], reverse=True)[:100]}
    # print(f"h: {h}")
    # analyze_fragment_detection(fragment_detection("./tests/files/t5-corpus/t5/*.text", "./uncompressed_weights.db", 20, [1, 5, 10, 20, 30, 50, 60, 70, 80, 95], False))
    analyze_common_block_detection(common_block_detection(glob.glob("./tests/files/t5-corpus/t5/*.text"), [100, 66.66, 42.86, 25, 11.11, 5.2, 4.1, 3.09, 2.04, 1.01], 10, "./uncompressed_weights.db"))


if __name__ == '__main__':
    main()
