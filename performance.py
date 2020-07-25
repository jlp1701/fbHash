import os
import random
import math
from fbHash import fbHashB
import ssdeep
import glob
import numpy as np
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
        with open(fp, 'rb') as f:
            data = f.read()
            frag_data.append({'file_path': fp, 'data_fbHash': fbHashB.hashd(data, w_path), 'data_ssdeep': ssdeep.hash(data)})

    print(f"start hashing fragments")
    # compute fragments of all files and hash them
    for fr in frag_data:
        with open(fr['file_path'], "rb") as f:
            data = f.read()
            fragments = get_fragments(data, frag_sizes, random_pos)
            fr['fragments_fbHash'] = list(map(lambda d: fbHashB.hashd(d, w_path), fragments))
            fr['fragments_ssdeep'] = list(map(lambda d: ssdeep.hash(d), fragments))

    print(f"start comparing...")
    comp_res = []
    # do comparisons
    for fi in frag_data:
        fj_res = []
        for fj in frag_data:
            frag_res = []
            for frag in fj['fragments_fbHash']:
                frag_res.append(fbHashB.compare(fi['data_fbHash'], frag))
            fj_res.append(frag_res)
        comp_res.append(fj_res)

    comp_res_ssdeep = []
    # do comparisons
    for fi in frag_data:
        fj_res = []
        for fj in frag_data:
            frag_res = []
            for frag in fj['fragments_ssdeep']:
                frag_res.append(ssdeep.compare(fi['data_ssdeep'], frag))
            fj_res.append(frag_res)
        comp_res_ssdeep.append(fj_res)

    return {'results': [comp_res, comp_res_ssdeep], 'frag_sizes': frag_sizes}


def analyze_fragment_detection(fragment_detection):
    # calc match percentage and average score of each fragment size
    frag_sizes = fragment_detection['frag_sizes']
    res = fragment_detection['results'][0]
    res_ssdeep = fragment_detection['results'][1]

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
    print(f"frag_res: fbHash: {frag_res}")
    # calc avg score for all fragments
    avg_score = []
    for i_frag in range(len(frag_sizes)):
        score = []
        for i_f in range(len(res)):
            score.append(res[i_f][i_f][i_frag])  # get the genuine comp scores
        avg_score.append(sum(score) / len(score))

    print(f"ssdeep")
    # ssdeep
    # calc match percentage
    # for all fragment sizes
    frag_res_ssdeep = []
    for fsize in range(len(frag_sizes)):
        n_gen_comp = 0
        for i in range(len(res_ssdeep)):
            # get genuine comp result
            gen_comp = res_ssdeep[i][i][fsize]

            # get impostor res and sort them descending
            imp_comp = sorted([c[fsize] for c in res_ssdeep[i] if c != res_ssdeep[i][i]], reverse=True)

            print(f"fsize: {frag_sizes[fsize]}: gen_comp: {gen_comp}; imp_comp: {imp_comp}")
            if gen_comp > imp_comp[0]:
                n_gen_comp += 1
            # print(f"n_gen_comp: {n_gen_comp}; len(res): {len(res)}")
        frag_res_ssdeep.append(n_gen_comp / len(res_ssdeep) * 100.0)
    print(f"frag_res: ssdeep: {frag_res_ssdeep}")
    # calc avg score for all fragments
    avg_score_ssdeep = []
    for i_frag in range(len(frag_sizes)):
        score = []
        for i_f in range(len(res_ssdeep)):
            score.append(res_ssdeep[i_f][i_f][i_frag])  # get the genuine comp scores
        avg_score_ssdeep.append(sum(score) / len(score))

    # plot
    # print(f"frag_res: {frag_res}")
    # fig, ax = plt.subplots()
    # ax.plot(frag_sizes, frag_sizes, c='gray')
    # ax.plot(frag_sizes, avg_score, c='g')
    # ax.plot(frag_sizes, frag_res, '--', c='g')

    # ax.plot(frag_sizes, avg_score_ssdeep, c='b')
    # ax.plot(frag_sizes, frag_res_ssdeep, '--', c='b')

    # ax.set(xlabel='fragment sizes', ylabel='avg score',
    #        title='Fragment detection')
    # ax.grid()



    width = 0.35

    frag_sizes.reverse()
    avg_score.reverse()
    avg_score_ssdeep.reverse()
    frag_res.reverse()
    frag_res_ssdeep.reverse()

    x = np.arange(len(frag_sizes))
    fig, ax = plt.subplots()
    
    ax.plot(x, frag_res, '--', c='b')
    ax.plot(x, frag_res_ssdeep, '--', c='darkorange')

    ax.bar(x - width / 2, avg_score, width, label='fbHash')
    ax.bar(x + width / 2, avg_score_ssdeep, width, label='ssdeep')
    # ax.plot(frag_sizes, frag_res, c='r')
    ax.set_ylim([0, 101])
    ax.set_xticks(x)
    ax.set_xticklabels(list(map(round, frag_sizes)))
    ax.legend()
    ax.set(xlabel='fragment sizes', ylabel='avg score', title='Fragment detection')
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
        if len(sink_set) < 2:
            print(f"Not enouth files available as sink!")
            continue
        f_sinks = random.sample(sink_set, 2)
        print(f"f_sinks: {f_sinks}")

        with open(f_src[0], 'rb') as f:
            data_src = f.read()
        # select two files for fragment sink
        with open(f_sinks[0][0], 'rb') as f:
            data_sinks[0] = f.read()
        with open(f_sinks[1][0], 'rb') as f:
            data_sinks[1] = f.read()

        # for each fragment size:
        res = {'fbHash': {}, 'ssdeep': {}}
        for frag_size in frag_sizes:
            # extract fragment from fragment source
            (data_frag, ) = get_fragments(data_src, [frag_size], random_pos=True)

            # insert fragment data into sinks at random positions
            data_sinks_ins = [data_sinks[0], data_sinks[1]]
            rnd_pos = random.randint(0, len(data_sinks[0]))
            # print(f"data_sinks[0]: {type(data_sinks[0])}")
            # print(f"data_frag: {type(data_frag)}")
            # data_sinks_ins[0][rnd_pos:rnd_pos] = data_frag
            data_sinks_ins[0] = data_sinks_ins[0][0:rnd_pos] + data_frag + data_sinks_ins[0][rnd_pos:]
            rnd_pos = random.randint(0, len(data_sinks[1]))
            # data_sinks_ins[1][rnd_pos:rnd_pos] = data_frag
            data_sinks_ins[1] = data_sinks_ins[1][0:rnd_pos] + data_frag + data_sinks_ins[1][rnd_pos:]

            # hash and compare the two sinks
            hash_sink = [fbHashB.hashd(data_sinks_ins[0], w_path), fbHashB.hashd(data_sinks_ins[1], w_path)]
            sim = fbHashB.compare(hash_sink[0], hash_sink[1])

            # normalize the result with respect to the fragment size
            # sim_n =  (frag_size / (frag_size + 100.0)) * 100 / sim
            print(f"frag_size: {frag_size}: sim[fbHash]: {sim}")

            # save result in list
            if frag_size not in res['fbHash']:
                res['fbHash'][frag_size] = [sim]
            else:
                res['fbHash'][frag_size].append(sim)

            # ssdeep
            hash_sink = [ssdeep.hash(bytes(data_sinks_ins[0])), ssdeep.hash(bytes(data_sinks_ins[1]))]
            sim = ssdeep.compare(hash_sink[0], hash_sink[1])
            print(f"frag_size: {frag_size}: sim[ssdeep]: {sim}")
            # save result in list
            if frag_size not in res['ssdeep']:
                res['ssdeep'][frag_size] = [sim]
            else:
                res['ssdeep'][frag_size].append(sim)

    # generate average result
    for frag_size in res['fbHash']:
        res['fbHash'][frag_size] = sum(res['fbHash'][frag_size]) / len(res['fbHash'][frag_size])

    for frag_size in res['ssdeep']:
        res['ssdeep'][frag_size] = sum(res['ssdeep'][frag_size]) / len(res['ssdeep'][frag_size])

    return res


def analyze_common_block_detection(results):
    # calc match percentage and average score of each fragment size
    # plot

    frag_sizes = list(map(lambda t: (t / (t + 100.0) * 100), results['fbHash']))
    avg_score_fbHash = list(map(lambda t: results['fbHash'][t], results['fbHash']))
    avg_score_ssdeep = list(map(lambda t: results['ssdeep'][t], results['ssdeep']))
    # frag_sizes.reverse()
    # avg_score_fbHash.reverse()
    # avg_score_ssdeep.reverse()
    print(f"frag_sizes: {frag_sizes}")
    print(f"avg_score_fbHash: {avg_score_fbHash}")
    print(f"avg_score_ssdeep: {avg_score_ssdeep}")

    width = 0.35

    x = np.arange(len(frag_sizes))
    fig, ax = plt.subplots()
    # ax.plot(frag_sizes, frag_sizes, c='k')
    ax.bar(x - width / 2, avg_score_fbHash, width, label='fbHash')
    ax.bar(x + width / 2, avg_score_ssdeep, width, label='ssdeep')
    # ax.plot(frag_sizes, frag_res, c='r')
    ax.set_ylim([0, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(list(map(round, frag_sizes)))
    ax.legend()

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
    analyze_fragment_detection(fragment_detection("./tests/files/t5-corpus/t5/*.text", "./weights_1000_no_xls_doc_jpg.db", 5, [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], True))
    # analyze_common_block_detection(common_block_detection(glob.glob("./tests/files/t5-corpus/t5/*.text"), [100, 66.66, 42.86, 25, 11.11, 5.2, 4.1, 3.09, 2.04, 1.01], 5, "./weights_1000_no_xls_doc_jpg.db"))


if __name__ == '__main__':
    main()
