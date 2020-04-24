import io
import math
from collections import deque


class FbHashB(object):
	"""docstring for FbHashB"""
	def __init__(self):
		super(FbHashB, self).__init__()
		# make sure that there is precomputed list of document weights from a cluster of reference documents (NIST)
		# maybe load list of document weights from file instead of hard code them

def hash(file_path, doc_weights):
	# compute chunk freqency of document
	chunks = compute_chunk_freq(file_path)

	# (normalize chunk frequencies) normalization is applying the logarithm
	for ch in chunks:
		doc_w = 1
		if ch in doc_weights:
			doc_w = doc_weights[ch]
		# compute chunk weight and then score
		chunks[ch] = (1 + math.log10(chunks[ch])) * doc_w
	return chunks

def compute_chunk_freq(file_path):
	# compute rolling hash for every byte in document
	chunk_list = get_chunks(file_path)

	# increment occurrence of chunk in hashmap
	ch_freq_dict = {}
	for ch in chunk_list:
		if not ch in ch_freq_dict:
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
	# for each document calculate the unique set of chunks	
	for doc in ref_docs:
		chunk_set = get_unique_chunks(get_chunks(doc))
		# increment the hashmap entries for these chunks
		for ch in chunk_set:
			if not ch in doc_freq_dict:
				doc_freq_dict[ch] = 1
			else:
				doc_freq_dict[ch] += 1	

	# calculate document weights
	for ch in doc_freq_dict:
		doc_freq_dict[ch] = math.log10(N/doc_freq_dict[ch])
	return doc_freq_dict

def get_chunks(file_path):
	data = []
	chunk_list = []
	r_hash = RollingHash()
	with open(file_path, "rb") as f:
		data = list(f.read())  ## read in file in one go

	#print(f"file length: {len(data)}")
	if len(data) < r_hash.k:
		raise f"File too small. Must contain at least {r_hash.k} bytes."
	
	# compute rolling hash for complete file
	# read in k-1 bytes to build the first hash value
	for i in range(r_hash.k-1):
		r_hash.digest_byte(data.pop(0))

	#print("hash data ...")
	for b in data:
		r_hash.digest_byte(b)		
		d = r_hash.get_digest()
		chunk_list.append(d)
	
	return chunk_list

def get_unique_chunks(chunks):
	return set(chunks)

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

		# digest = digest * a + byte mod n
		self.digest = (self.digest * self.a + byte) % self.n

		# if deque size is > k: pop value from deque and compute:
		if len(self.items) > self.k:
			# digest = digest - value * a ** k mod n
			value = self.items.popleft()
			#print(f"value popped: {value}")
			self.digest = (self.digest - value * self.a ** self.k) % self.n
		#print(f"values: {self.items}")

	def get_digest(self):
		return self.digest

		

		