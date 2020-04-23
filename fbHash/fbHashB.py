from collections import deque


class FbHashB(object):
	"""docstring for FbHashB"""
	def __init__(self):
		super(FbHashB, self).__init__()
		# make sure that there is precomputed list of document weights from a cluster of reference documents (NIST)
		# maybe load list of document weights from file instead of hard code them

	def hash(self, file_path):
		# compute chunk freqency of document

		# (normalize chunk frequencies) normalization is applying the logarithm

		# compute chunk weights

		# compute chunk scores

		# return list of chunk scores (maybe as a sparse array)
		pass

def compute_chunk_freq(file_path):
	# compute rolling hash for every byte in document
	# increment occurrence of chunk in hashmap
	# return hashmap
	pass

def compare(hash_1, hash_2):
	# when using sparse vectors, this is easy as multiplying vectors is easy with numpy
	pass

def compute_document_weights(ref_docs):
	# for each document calculate the unique set of chunks

	# increment the hashmap entries for these chunks

	# return hashmap
	pass

def get_chunk_set(file_path):
	# compute rolling hash for complete file
	# check if calculated chunk is already known
	# if not, add chunk to list
	# return list
	pass


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

		

		