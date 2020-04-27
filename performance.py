import random
import math
from fbHash import fbHashB
import glob


# function for selecting random test files (implemented in helper.py)

# function to generate fragment (sequential or random) from file
def get_fragments(data, sizes, random_pos=False):
	frag = []
	random.seed()
	for s in sizes:
		l = math.ceil(len(data) * s / 100.0)
		d = 0
		if random_pos:
			d = random.randint(0, len(data)-l)
		frag.append(data[d:d+l-1])
	return frag

def fragment_detection(dir, w_path, num_files, frag_sizes, random_pos):
	# set up pairings and comparisons to be computed by several tools
	# for all tools:
		# perform hashing of all files and fragments for all tools
		# evaluate comparisons (1) match percentage (2) average similarity score 
	frag_data = []
	files = random.sample(glob.glob(dir), num_files)

	# read in all files
	for fp in files:
		frag_data.append({'file_path':fp, 'data':fbHashB.hashf(fp, w_path)})

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

	return comp_res
	# store comparisons



def main():
	print(fragment_detection("./tests/files/t5-corpus/t5/**", "./uncompressed_weights.db", 2, [10, 95], False))

if __name__ == '__main__':
	main()