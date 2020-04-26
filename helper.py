import re
import glob
import json
import bigjson
from fbHash import fbHashB

def doc_weights_from_dir(dir_path):
	# get list of all files in dir matching regex
	files = glob.glob(dir_path, recursive=True)
	print(f"Number of files: {len(files)}")

	# generate doc_weights
	weights = fbHashB.compute_document_weights(files)
	return weights

def doc_weights2json(weights, file_path):
	with open(file_path, "w") as f:
		for ch in weights:
			f.write(f"{ch},{weights[ch]}\n")

def json2doc_weights(file_path):
	w = {}
	with open(file_path, "r") as f:
		while True:
			line = f.readline()
			if not line:
				break
			ch,val = line.split(',')
			w[int(ch)] = float(val)
	return w

def select_ref_files(dir_path, num):
	# sort files according to their type (file ending)
	files = glob.glob(dir_path)

	if len(files) < num:
		raise Exception(f"Not enough files in path. Found {len(files)} files.")

	# get all file endings
	r = re.compile(".*\\.(\\w+)$")  # captures file endings
	types = set([r.match(m).group(1) for m in files if r.match(m)])

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
		if (len(files_sorted[i%len(files_sorted)]) > 0):
			picked.append(files_sorted[i%len(files_sorted)].pop(0))
		i += 1
	return picked

def pick_and_gen_doc_weights(file_path, num, save_path):
	ref_files = select_ref_files(file_path, num)
	print("Generate weights from files:")
	print(ref_files)
	w = fbHashB.compute_document_weights(ref_files)
	print(f"number of weights: {len(w)}")
	#for k in (sorted(w, key=w.get, reverse=True)[:10]):
	#	print(f"{k}: {w[k]}")
	doc_weights2json(w, save_path)
	print("serialized")

	# load and check serialized weights with generated
	#w_loaded = json2doc_weights(save_path)
	#print(f"num loaded from file: {len(w_loaded)}")
	#if len(w.keys()) != len(w_loaded.keys()):
	#	raise Exception("Number of weights is not equal.")
	#for ch in w:
	#	if w[ch] != w_loaded[ch]:
	#		raise Exception("Weights are not equal.")

def main():
	pick_and_gen_doc_weights("./tests/files/t5-corpus/t5/**", 1000, "weights.json")

if __name__ == '__main__':
	main()