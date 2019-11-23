import numpy as np
import pdb, csv
import matplotlib.pyplot as plt

data = np.load('../../panam/Life Code Ground Truth/life_affect+content.npy', allow_pickle=True).item()
data['family'] = [int(fid) for fid in data['family']]

mother = {}
child = {}
constructs = ['Aggressive', 'Dsyphoric', 'Positive', 'Other']
constructs_labels = ['Aggressive(M)', 'Dsyphoric(M)', 'Positive(M)', 'Other(M)', 'Aggressive(C)', 'Dsyphoric(C)', 'Positive(C)', 'Other(C)']

midx = np.where(np.array(data['subject'])=='2')
cidx = np.where(np.array(data['subject'])=='1')
# print(np.unique(data['construct_code']))


# for key in data.keys():
	# mother[key] = np.array(data[key])[midx]

# for key in data.keys():
	# child[key] = np.array(data[key])[cidx]

def get_depressed_families(file = '/home/mab623/panam/2019.8.7_Multimodal TPOT Data.csv'):
	depressed_list = []
	fp = open(file, 'r')
	csvreader = csv.reader(fp)
	for row in csvreader:
		# print(row)
		if row[1] == '1':
			depressed_list.append(int(row[0][-4:]))
			# print(row[0])
	# print(len(depressed_list))
	return depressed_list

def calculate_transistions(families):
	'''
	Find the effect of sub2 on sub1
	'''
	transition_matrix = np.zeros((8, 8))
	print('using {0} families'.format(len(families)))

	_fps = 29.97
	window = 2*_fps # 2 sec window
	confusion_matrix = np.zeros((2, 4, 4))
	# pdb.set_trace()
	# print('family used')
	for family in families: # For each family
		# print(family)
		# pdb.set_trace()
		fam_annotations = np.where(np.array(data['family']) == family)[0]
		# print(min(fam_annotations))
		# for key in data.keys(): # get the family specific data
			# _fam[key] = [data[key][fid].astype(np.float32) for fid in fam_annotations]

		# pdb.set_trace()
		for idx in range(len(fam_annotations)-1): #data['construct_code'][min(fam_annotations):max(fam_annotations)-1]):
			curr_con = data['construct_code'][fam_annotations[0]+idx]
			next_con = data['construct_code'][fam_annotations[0]+idx+1]
			print(data['onset'][fam_annotations[0]+idx], data['subject'][fam_annotations[0]+idx], data['construct_code'][fam_annotations[0]+idx], data['affect_code'][fam_annotations[0]+idx], data['content_code'][fam_annotations[0]+idx])
			# if curr_con == next_con and data['subject'][fam_annotations[0]+idx] == data['subject'][fam_annotations[0]+idx+1] and curr_con!=3:
			# 	# pdb.set_trace()
			# 	print(family, data['subject'][fam_annotations[0]+idx])
			if data['subject'][fam_annotations[0]+idx] == '1':
				row_offset = 4
			else:
				row_offset = 0

			if data['subject'][fam_annotations[0]+idx+1] == '1':
				col_offset = 4
			else:
				col_offset = 0

			transition_matrix[row_offset+curr_con, col_offset+next_con] += 1

	return transition_matrix

def plot_cm(cm):
	
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest')
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),
		# ... and label them with the respective list entries
		xticklabels=constructs_labels, yticklabels=constructs_labels,
		title='Transition matrix',
		ylabel='Next state',
		xlabel='Current state')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f'# if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.show()

	return ax

def	normalized_cm(conf_mat):

	# pdb.set_trace()
	_divisor = np.sum(conf_mat, axis=1)
	conf_mat = np.divide(conf_mat, _divisor[:, np.newaxis])

	return conf_mat#mother_mother, mother_child, child_mother, child_child


if __name__ == '__main__':

	families = np.unique(np.array(data['family']))
	all_population = calculate_transistions(families)
	all_population_norm = normalized_cm(all_population)
	# print('All population \n', all_population_norm, np.sum(all_population, (0,1)))

	depressed_families = get_depressed_families()

	depressed_population = calculate_transistions(depressed_families)
	depressed_population_norm = normalized_cm(depressed_population)
	# print('All population \n', depressed_population_norm, np.sum(depressed_population, (0,1)))

	print(all_population, '\n', depressed_population)

	healthy_population = all_population - depressed_population
	healthy_population_norm = normalized_cm(healthy_population)

	with open('transistion.csv', 'w') as fp:
		csvwriter = csv.writer(fp)

		csvwriter.writerow(['All population'])
		for row in range(all_population_norm.shape[0]):
			csvwriter.writerow(all_population_norm[row, :])

		csvwriter.writerow(['\n'])

		csvwriter.writerow(['Depressed population'])
		for row in range(depressed_population_norm.shape[0]):
			csvwriter.writerow(depressed_population_norm[row, :])
		csvwriter.writerow(['\n'])


		csvwriter.writerow(['Healthy population'])
		for row in range(healthy_population_norm.shape[0]):
			csvwriter.writerow(healthy_population_norm[row, :])

	# final_confusion_matrix = np.concatenate((np.concatenate((mother_mother, mother_child), axis=1), np.concatenate((child_mother, child_child), axis=1)), axis=0)
	# plot_cm(final_confusion_matrix)

