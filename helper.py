import os
import re
import glob
import json
import bigjson
import sqlite3
import random
from fbHash import fbHashB

def doc_weights_from_dir(dir_path):
	# get list of all files in dir matching regex
	files = glob.glob(dir_path, recursive=True)
	print(f"Number of files: {len(files)}")

	# generate doc_weights
	weights = fbHashB.compute_document_weights(files)
	return weights


def select_ref_files(dir_path, num, exclude_type=[]):
	random.seed()
	# sort files according to their type (file ending)
	files = glob.glob(dir_path)

	if len(files) < num:
		raise Exception(f"Not enough files in path. Found {len(files)} files.")

	print(f"exclude_type: {exclude_type}")

	# get all file endings
	r = re.compile(".*\\.(\\w+)$")  # captures file endings
	types = set([r.match(m).group(1) for m in files if r.match(m)])
	types = types - set(exclude_type)

	print(f"types: {types}")

	files_sorted = []
	for t in types:
		f = [m for m in files if re.match(f".*\\.{t}$", m)]
		files_sorted.append(f)
	#print(f"files_sorted: {files_sorted}")
	if len(files_sorted) == 0:
		raise Exception(f"No files to select from.")

	# add one file from each bucket until num is reached
	picked = []
	i = 0
	while len(picked) < num:  # loop terminates because of check at beginning of func
		flist = files_sorted[i%len(files_sorted)]
		if (len(flist) > 0):
			item = random.choice(flist)
			picked.append(item)
			flist.remove(item)
		i += 1
	return picked

def pick_and_gen_doc_weights(file_path, num, save_path, exclude_type=[]):
	ref_files = select_ref_files(file_path, num, exclude_type)
	print("Generate weights from files:")
	print(ref_files)
	w = fbHashB.compute_document_weights(ref_files)
	print(f"number of weights: {len(w)}")
	#for k in (sorted(w, key=w.get, reverse=True)[:10]):
	#	print(f"{k}: {w[k]}")
	fbHashB.doc_weights2sqlite(w, save_path)
	print("serialized")

def main():
	pick_and_gen_doc_weights("./tests/files/t5-corpus/t5/**", 200, "uncompressed_weights.db", ["pdf", "ppt"])

if __name__ == '__main__':
	main()