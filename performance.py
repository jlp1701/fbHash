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
	print(f"files picked: {files}")

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

	return {'results': comp_res, 'frag_sizes': frag_sizes}

def analyze_fragment_detection(fragment_detection):
	# calc match percentage and average score of each fragment size
	frag_sizes = fragment_detection['frag_sizes']
	res = fragment_detection['results']

	#print(f"res: {res}")
	# calc match percentage
	# for all fragment sizes
	frag_res = []
	for fsize in range(len(frag_sizes)):
		n_gen_comp = 0
		for i in range(len(res)):
			# get genuine comp result
			gen_comp = res[i][i][fsize]

			# get impostor res and sort them descending
			imp_comp  = sorted([c[fsize] for c in res[i] if c != res[i][i]], reverse=True)
			
			print(f"fsize: {frag_sizes[fsize]}: gen_comp: {gen_comp}; imp_comp: {imp_comp}")
			if gen_comp > imp_comp[0]:
				n_gen_comp += 1
			#print(f"n_gen_comp: {n_gen_comp}; len(res): {len(res)}")
		frag_res.append(n_gen_comp / len(res) * 100.0)
	

	# calc avg score for all fragments
	avg_score = []
	for i_frag in range(len(frag_sizes)):
		score = []
		for i_f in range(len(res)):
			score.append(res[i_f][i_f][i_frag]) # get the genuine comp scores
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

	#fig.savefig("test.png")
	plt.show()

def common_block_detection(frag_data, test_set, frag_sizes, w_path):
	# for all fragment sizes:
		# generate fragment data
		# insert into test_set data
		# do a pairwise comparison of the two test files
	pass

def common_block_multi(dir, runs, frag_sizes, w_path):
	# for number of runs:
		# select three files from pool and remove them
		# do common block detection and save results
	pass

def analyze_common_block_detection(results):
	# calc match percentage and average score of each fragment size
	pass

def main():
	with open("./tests/files/t5-corpus/t5/004958.text", "rb") as f:
		d = list(f.read())
	h = {k : v for k, v in sorted(fbHashB.compute_chunk_freq(d).items(), key=lambda item: item[1], reverse=True)[:100]}
	print(f"h: {h}")
	analyze_fragment_detection(fragment_detection("./tests/files/t5-corpus/t5/*.ppt", "./uncompressed_weights.db", 20, [1, 5, 10, 20, 30, 50, 60, 70, 80, 95], False))

if __name__ == '__main__':
	main()