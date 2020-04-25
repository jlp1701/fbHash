import pytest
from fbHash import fbHashB

def test_test():
	assert True == True

def test_hashing():
	file_path = "./files/testfile_1.txt"
	pass

def test_document_weights():
	files = ["./tests/files/testfile_1.txt", "./tests/files/testfile_2.txt", "./tests/files/testfile_3.txt", "./tests/files/testfile_4.txt"]
	doc_w = fbHashB.compute_document_weights(files)
	assert len(doc_w) > 0

def test_chunk_freq():

	ch_fr1 = fbHashB.compute_chunk_freq("tests/files/testfile_1.txt")
	ch_fr1_1 = fbHashB.compute_chunk_freq("tests/files/testfile_1_1.txt")
	ch_fr2 = fbHashB.compute_chunk_freq("tests/files/testfile_2.txt")
	ch_fr3 = fbHashB.compute_chunk_freq("tests/files/testfile_3.txt")
	ch_fr4 = fbHashB.compute_chunk_freq("tests/files/testfile_4.txt")

	assert len(ch_fr1.keys()) > 0
	assert len(ch_fr1_1.keys()) > 0
	assert len(ch_fr2.keys()) > 0
	assert len(ch_fr3.keys()) > 0
	assert len(ch_fr4.keys()) > 0

	# different files
	assert len(ch_fr1.keys() & ch_fr2.keys()) == 0

	## one common chunk
	assert len(ch_fr1.keys() & ch_fr1_1.keys()) == 1

def test_comparison():
	files = ["./tests/files/testfile_1.txt", "./tests/files/testfile_1_1.txt", "./tests/files/testfile_2.txt", "./tests/files/testfile_3.txt"]
	doc_w = fbHashB.compute_document_weights(files)
	h1 = fbHashB.hash(files[0], doc_w)
	h1_1 = fbHashB.hash(files[1], doc_w)
	h2 = fbHashB.hash(files[2], doc_w)

	# different files
	assert fbHashB.compare(h1, h2) == 0

	# similar files
	assert 40 < fbHashB.compare(h1, h1_1) < 60