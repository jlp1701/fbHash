import pytest
from fbHash import fbHashB

def test_test():
	assert True == True

def test_assumptions():
	r_hash = fbHashB.RollingHash()
	assert 255*255**(r_hash.k-1) <= 2**64-1
	assert 255*255**(r_hash.k-1) < r_hash.n

def test_hashes():
	# null hash
	r_hash = fbHashB.RollingHash()
	assert r_hash.get_digest() == 0

	# single byte hash
	r_hash.digest_byte(0)
	assert r_hash.get_digest() == 0

	# null series hash
	for i in range(100):
		r_hash.digest_byte(0)
	assert r_hash.get_digest() == 0

	# pre-calculated hash digest of bytes: 1..7
	r_hash = fbHashB.RollingHash()
	for b in range(1,8):
		r_hash.digest_byte(b)
		print(f"digest after {b}: {r_hash.get_digest()}")
	assert r_hash.get_digest() == 277111156113412
	r_hash.digest_byte(8)
	assert r_hash.get_digest() == 553135601810693
	r_hash.digest_byte(9)
	assert r_hash.get_digest() == 829160047507974

	# pre-calculated hash digest of bytes: 0..255
	r_hash = fbHashB.RollingHash()
	for b in range(0,256):
		r_hash.digest_byte(b)
		print(f"digest after {b}: {r_hash.get_digest()}")
	assert r_hash.get_digest() == 68731173689039100

