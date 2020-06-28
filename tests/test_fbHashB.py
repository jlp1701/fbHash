import pytest
from fbHash import fbHashB


def read_file(file_path):
    with open(file_path, "rb") as f:
        return list(f.read())


def test_test():
    assert True == True


def test_document_weights():
    files = ["./tests/files/testfile_1.txt", "./tests/files/testfile_2.txt", "./tests/files/testfile_3.txt", "./tests/files/testfile_4.txt"]
    doc_w = fbHashB.compute_document_weights(files)
    assert len(doc_w) > 0


def test_chunk_freq():
    d1 = read_file("tests/files/testfile_1.txt")
    d1_1 = read_file("tests/files/testfile_1_1.txt")
    d2 = read_file("tests/files/testfile_2.txt")
    d3 = read_file("tests/files/testfile_3.txt")
    d4 = read_file("tests/files/testfile_4.txt")

    ch_fr1 = fbHashB.compute_chunk_freq(d1)
    ch_fr1_1 = fbHashB.compute_chunk_freq(d1_1)
    ch_fr2 = fbHashB.compute_chunk_freq(d2)
    ch_fr3 = fbHashB.compute_chunk_freq(d3)
    ch_fr4 = fbHashB.compute_chunk_freq(d4)

    assert len(ch_fr1.keys()) == 1
    assert len(ch_fr1_1.keys()) == 2
    assert len(ch_fr2.keys()) == 1
    assert len(ch_fr3.keys()) == 1
    assert len(ch_fr4.keys()) == 187

    # different files
    assert len(ch_fr1.keys() & ch_fr2.keys()) == 0

    # one common chunk
    assert len(ch_fr1.keys() & ch_fr1_1.keys()) == 1


def test_unique_chunks():
    assert len(fbHashB.get_chunks(read_file("tests/files/testfile_1.txt"))) == 1
    assert len(fbHashB.get_unique_chunks("tests/files/testfile_1.txt")) == 1

    assert len(fbHashB.get_chunks(read_file("tests/files/testfile_1_2.txt"))) == 27
    assert len(fbHashB.get_unique_chunks("tests/files/testfile_1_2.txt")) == 1


def test_comparison():
    files = ["./tests/files/testfile_1.txt", "./tests/files/testfile_1_1.txt", "./tests/files/testfile_2.txt", "./tests/files/testfile_3.txt"]
    doc_w_path = "test_weights.db"
    doc_w = fbHashB.compute_document_weights(files)
    fbHashB.doc_weights2sqlite(doc_w, doc_w_path)
    h1 = fbHashB.hashf(files[0], doc_w_path)
    h1_1 = fbHashB.hashf(files[1], doc_w_path)
    h2 = fbHashB.hashf(files[2], doc_w_path)

    # different files
    assert fbHashB.compare(h1, h2) == 0

    # similar files
    assert 40 < fbHashB.compare(h1, h1_1) < 60
