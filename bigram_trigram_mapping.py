def map_affect_codes(data):

	affect_mapping = {'0':0, '1':0, '2':2, '3':2, '4':1, '5':1, '6':1, '7':1, '8':2, '9':3}

	for idx, affect in enumerate(data['affect_code']):
		data['affect_code'][idx] = affect_mapping[str(affect)]

	# swap affect and code data, so you dont need to update the bigram-trigram code
	data['affect_code'], data['construct_code'] = data['construct_code'], data['affect_code']

	return data


def bigram_mapping(criterion='construct'):

	idx = 0
	bigram_mapping = {}

	if criterion == 'construct':
		for sub1 in range(1, 3): # mother or child
			for sub2 in range(1, 3): # mother or child
				for curr_construct in range(0, 4): # considering all 4 constructs
					for next_construct in range(0, 4): # considering all 4 constructs

						bigram_mapping[str(curr_construct)+str(next_construct)+str(sub1)+str(sub2)] = idx
						idx+=1
	else:
		for sub1 in range(1, 3):  # mother or child
			for sub2 in range(1, 3):  # mother or child
				for curr_affect in range(0, 10):  # considering all 4 constructs
					for next_affect in range(0, 10):  # considering all 4 constructs
						bigram_mapping[str(curr_affect) + str(next_affect) + str(sub1) + str(sub2)] = idx
						idx += 1

	print('max combinations-{0} for {1} based histograms'.format(idx, criterion))
	return bigram_mapping

def trigram_mapping(criterion='construct'):

	idx = 0
	trigram_mapping = {}

	if criterion == 'construct':
		for sub1 in range(1, 3): # mother or child
			for sub2 in range(1, 3): # mother or child
				for sub3 in range(1, 3):
					for t0_construct in range(0, 4): # considering all 4 constructs
						for t1_construct in range(0, 4): # considering all 4 constructs
							for t2_construct in range(0, 4): # considering all 4 constructs
								trigram_mapping[str(t0_construct)+str(t1_construct)+str(t2_construct)+str(sub1)+str(sub2)+str(sub3)] = idx
								idx+=1

	else:
		for sub1 in range(1, 3):  # mother or child
			for sub2 in range(1, 3):  # mother or child
				for sub3 in range(1, 3):
					for t0_affect in range(0, 10):  # considering all 4 constructs
						for t1_affect in range(0, 10):  # considering all 4 constructs
							for t2_affect in range(0, 10):  # considering all 4 constructs
								trigram_mapping[
									str(t0_affect) + str(t1_affect) + str(t2_affect) + str(sub1) + str(
										sub2) + str(sub3)] = idx
								idx += 1

	print('max combinations-{0} for {1} based histograms'.format(idx, criterion))
	return trigram_mapping


if __name__ == '__main__':
	_map = trigram_mapping()
	print(_map)
	# print(_map[str(0)+str(0)+str(1)+str(2)]) # bigram test
	print(_map[str(0)+str(0)+str(0)+str(2)+str(1)+str(2)])  # trigram test
