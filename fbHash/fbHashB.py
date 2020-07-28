import os
import io
import math
import sqlite3
from collections import deque
from multiprocessing import Pool


def get_weights(file_path, chunks):
    conn = sqlite3.connect(file_path)
    c = conn.cursor()
    w = {}
    s = []
    n = 30
    keys = list(chunks.keys())
    for i in range(0, len(keys), n):
        li = keys[i:i + n]
        c.execute(f"SELECT chunk, weight FROM docweights WHERE chunk IN ({', '.join(['?' for _ in li])})", li)
        res = c.fetchall()
        # print(f"res: {res}")
        for (k, v) in res:
            w[k] = v

    conn.commit()
    conn.close()
    return w


def hashd_weight_file(data, doc_w_file):
    # compute chunk freqency of document
    chunks = compute_chunk_freq(data)
    weights = get_weights(doc_w_file, chunks)

    # (normalize chunk frequencies) normalization is applying the logarithm
    for ch in chunks:
        if ch not in weights:
            w = 1.0
        else:
            w = weights[ch]
        chunks[ch] = (1 + math.log10(chunks[ch])) * w
    return chunks


def hashd_weights(data, weights):
    # compute chunk freqency of document
    chunks = compute_chunk_freq(data)

    # (normalize chunk frequencies) normalization is applying the logarithm
    for ch in chunks:
        if ch not in weights:
            w = 1.0
        else:
            w = weights[ch]
        chunks[ch] = (1 + math.log10(chunks[ch])) * w
    return chunks


def hashf(file_path, doc_w_file):
    with open(file_path, "rb") as f:
        data = f.read()
    return hashd_weight_file(data, doc_w_file)


def compute_chunk_freq(data):
    # compute rolling hash for every byte in document
    chunk_list = get_chunks(data)

    # increment occurrence of chunk in hashmap
    ch_freq_dict = {}
    for ch in chunk_list:
        if ch not in ch_freq_dict:
            ch_freq_dict[ch] = 1
        else:
            ch_freq_dict[ch] += 1
    # return hashmap
    return ch_freq_dict


def compare(hash_1, hash_2):
    common_chunks = hash_1.keys() & hash_2.keys()
    ch_mul = 0
    for ch in common_chunks:
        ch_mul += hash_1[ch] * hash_2[ch]

    sqrt_sum1 = 0
    sqrt_sum2 = 0
    for ch in hash_1:
        sqrt_sum1 += hash_1[ch]**2
    sqrt_sum1 = math.sqrt(sqrt_sum1)

    for ch in hash_2:
        sqrt_sum2 += hash_2[ch]**2
    sqrt_sum2 = math.sqrt(sqrt_sum2)

    sim = ch_mul / (sqrt_sum1 * sqrt_sum2) * 100
    return sim


def compute_document_weights(ref_docs):
    doc_freq_dict = {}
    N = len(ref_docs)

    p_chunks = []
    with Pool(8) as p:
        p_chunks = p.map(get_unique_chunks, ref_docs)
    # print(f"num chunk sets: {len(p_chunks)}")

    # for each document calculate the unique set of chunks
    for ch_set in p_chunks:
        # increment the hashmap entries for these chunks
        for ch in ch_set:
            if ch not in doc_freq_dict:
                doc_freq_dict[ch] = 1
            else:
                doc_freq_dict[ch] += 1

    # calculate document weights
    for ch in doc_freq_dict:
        doc_freq_dict[ch] = math.log10(N / doc_freq_dict[ch])
    return doc_freq_dict


def create_doc_db(file_path):
    conn = sqlite3.connect(file_path)
    c = conn.cursor()
    c.execute("CREATE TABLE docweights (chunk INTEGER PRIMARY KEY ASC, weight REAL)")
    conn.commit()
    conn.close()


def doc_weights2sqlite(weights, file_path):
    # check if file exists and truncate
    if os.path.isfile(file_path):
        os.remove(file_path)

    create_doc_db(file_path)
    conn = sqlite3.connect(file_path)
    c = conn.cursor()
    i = 0
    for ch in weights:
        c.execute("INSERT INTO docweights VALUES (?,?)", [ch, weights[ch]])
        i += 1
        if i % 500000 == 0:
            conn.commit()

    conn.commit()
    conn.close()


def get_chunks(data):
    chunk_list = []
    r_hash = RollingHash()

    # print(f"file length: {len(data)}")
    if len(data) < r_hash.k:
        raise Exception(
            f"File too small. Must contain at least {r_hash.k} bytes.")

    # compute rolling hash for complete file
    # read in k-1 bytes to build the first hash value
    for b in data[:r_hash.k - 1]:
        r_hash.digest_byte(b)

    # print("hash data ...")
    for b in data[r_hash.k - 1:]:
        r_hash.digest_byte(b)
        d = r_hash.get_digest()
        chunk_list.append(d)

    return chunk_list


def get_unique_chunks(doc):
    # data = []
    with open(doc, "rb") as f:
        data = f.read()
    return set(get_chunks(data))


class RollingHash(object):
    """docstring for RollingHash"""

    def __init__(self):
        super(RollingHash, self).__init__()
        self.items = deque()
        self.k = 7
        self.n = 2305843009213693951  # mersenne prime: 2^p - 1; with p = 61
        self.a = 255
        self.digest = 0

    def digest_byte(self, byte):
        # push byte into deque
        self.items.append(byte)

        d = (self.digest * self.a + byte)

        # if deque size is > k: pop value from deque and compute:
        if len(self.items) > self.k:
            value = self.items.popleft()
            d = (d - value * self.a ** self.k)
        if d >= self.n:
            raise Exception(f"Computed larger number than n: {d}. window: {self.items}")
        self.digest = d

    def get_digest(self):
        return self.digest
