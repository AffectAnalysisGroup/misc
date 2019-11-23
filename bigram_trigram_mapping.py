
def bigram_mapping():

	idx = 0
	bigram_mapping = {}


	for sub1 in range(1, 3): # mother or child
		for sub2 in range(1, 3): # mother or child
			for curr_construct in range(0, 4): # considering all 4 constructs 
				for next_construct in range(0, 4): # considering all 4 constructs 

					bigram_mapping[str(curr_construct)+str(next_construct)+str(sub1)+str(sub2)] = idx
					idx+=1

	print('max combinations-{0}'.format(idx))
	return bigram_mapping

def trigram_mapping():

	idx = 0
	trigram_mapping = {}

	for sub1 in range(1, 3): # mother or child
		for sub2 in range(1, 3): # mother or child
			for sub3 in range(1, 3):
				for t0_construct in range(0, 4): # considering all 4 constructs 
					for t1_construct in range(0, 4): # considering all 4 constructs 
						for t2_construct in range(0, 4): # considering all 4 constructs
							trigram_mapping[str(t0_construct)+str(t1_construct)+str(t2_construct)+str(sub1)+str(sub2)+str(sub3)] = idx
							idx+=1

	print('max combinations-{0}'.format(idx))
	return trigram_mapping


if __name__ == '__main__':
	_map = trigram_mapping()
	print(_map)
	# print(_map[str(0)+str(0)+str(1)+str(2)]) # bigram test
	print(_map[str(0)+str(0)+str(0)+str(2)+str(1)+str(2)])  # trigram test
