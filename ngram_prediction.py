# Bi-gram and Tri-gram analysis of the construct transitions

# TODO add labels- PHQ-9 and groups

import numpy as np
import csv, pdb, pickle
from bigram_trigram_mapping import bigram_mapping, trigram_mapping
from life_transitions_redone import get_depressed_families
import pandas as pd
from sklearn import svm
from model_def import Classifier

import warnings
warnings.filterwarnings('ignore')

bigram_mapping_ = bigram_mapping()
trigram_mapping_ = trigram_mapping()
LAG_LIMIT = 7
LAG_ANALYSIS = True

# families that did not consent; IDs in ORI's notation
non_consent = ['0757', '0825', '1487', '1666', '2620', '2630', '2643', '4181', '5075', '5701', '9785']
# revisted families may begin with 10XXXX or 11XXXX but only 11XXXX should be considered
revisited_ids = ['0167', '0537', '0568', '0606', '0609','0788','0906', '1053', '1118']
data = np.load('../../panam/Life Code Ground Truth/life_affect+content.npy', allow_pickle=True).item()
# remove unconsented families and consider the revisted families
data['family'] = [int('11'+fid) if fid in revisited_ids else int('10'+fid) for fid in data['family'] if fid not in non_consent]

classification_data_trigram = []
classification_data_bigram = []
bigram_histogram_all_families = np.zeros(len(bigram_mapping_.keys()))
trigram_histogram_all_families = np.zeros(len(trigram_mapping_.keys()))

df = pd.read_csv('/home/mab623/panam/2019.8.7_Multimodal TPOT Data.csv', usecols = ['FamId', 'M_3L1TOT', 'GROUP'], dtype={'M_3L1TOT':np.float32})
fam_phq_group = df.to_numpy()[:169]

df = pd.read_excel('/home/mab623/panam/2019.10.23_TPOTDepressionScoresInventoryKW.xlsx', usecols = ['TPOT FamId', 'Lag between Mother PH9Q and Visit Date'])
tpot_lag = df.to_numpy()[:169]
# bigram transitions

lag_families = tpot_lag[np.intersect1d(np.where(-1*LAG_LIMIT <= tpot_lag[:, 1]), np.where(tpot_lag[:, 1]<=LAG_LIMIT)), :]

def sanity_check():
	'''
	Do sanity check on the .sds file that ORI sent for repetitions of the codes
	'''
	affect = np.zeros(10)
	content = np.zeros(100)

	with open('../../panam/2019.9.25_alldata.sds') as fp:
		prev_line = ''
		err = 0
		msub = 0
		csub = 0
		fam_id = {}

		for lid, line in enumerate(fp):

			if lid > 22:
				if line[0] == '<' and line[1:-2] not in fam_id:
					curr_fam = line[1:-2]
					fam_id[line[1:-2]] = []
					err_lines = []

				else:
					curr_line, curr_time = line.rstrip('\n').split(',')

					if curr_line == prev_line:
						affect[int(curr_line[-1])] +=1
						content[int(curr_line[1:3])] +=1
						err+=1
						if curr_line[0] == '1':
							msub +=1
						else:
							csub +=1
						err_lines.append(curr_time)
					prev_line = curr_line

				fam_id[curr_fam] = err_lines


	print('errors-', err,'\n', err_lines, '\n', msub, csub, affect, content[np.where(content!=0)], np.where(content!=0)[0], fam_id)

	with open('repetition.csv', 'w') as f:
		csvwriter = csv.writer(f)
		csvwriter.writerow(['family', 'timestamps'])
		for key in fam_id.keys():
			csvwriter.writerow([key]+[fam_id[key]])
			# csvwriter.writerow([fam_id[key]])
	return

def calculate_bigram_histogram(families, depressed_families, depression_labels=True):
	'''
	'''
	print('using {0} families'.format(len(families)))
	labels = []

	for family in families: # For each family
		
		if depression_labels:
			if family in depressed_families:
				labels.append(1)
			else:
				labels.append(0)
		# else: # PHQ-9 scores


		bigram_hist = np.zeros(len(bigram_mapping_.keys()))
		fam_annotations = np.where(np.array(data['family']) == family)[0]
		for idx in range(len(fam_annotations)-1): #data['construct_code'][min(fam_annotations):max(fam_annotations)-1]):
			curr_con = data['construct_code'][fam_annotations[0]+idx]
			next_con = data['construct_code'][fam_annotations[0]+idx+1]
			sub_idx = data['subject'][fam_annotations[0]+idx]
			next_sub_idx = data['subject'][fam_annotations[0]+idx+1]
			_key = str(curr_con)+str(next_con)+str(sub_idx)+str(next_sub_idx)
			bigram_hist[bigram_mapping_[_key]]+=1
			bigram_histogram_all_families[bigram_mapping_[_key]]+=1

		classification_data_bigram.append(bigram_hist)
	
	return bigram_histogram_all_families, classification_data_bigram, labels

def calculate_trigram_histogram(families, depressed_families, depression_labels=True):
	'''
	Calculate the trigram histogram. 
	The histogram index is inferred from the trigram_mapping_ generted form bigram_trigram_mapping.py file
	'''
	print('using {0} families'.format(len(families)))
	labels = []

	for family in families: # For each family
		
		trigram_hist = np.zeros(len(trigram_mapping_.keys()))

		if depression_labels:
			if family in depressed_families:
				labels.append(1)
			else:
				labels.append(0)
		# else:


		fam_annotations = np.where(np.array(data['family']) == family)[0]
		for idx in range(len(fam_annotations)-2): #data['construct_code'][min(fam_annotations):max(fam_annotations)-1]):
			t0_con = data['construct_code'][fam_annotations[0]+idx]
			t1_con = data['construct_code'][fam_annotations[0]+idx+1]
			t2_con = data['construct_code'][fam_annotations[0]+idx+2]
			t0_sub_idx = data['subject'][fam_annotations[0]+idx]
			t1_sub_idx = data['subject'][fam_annotations[0]+idx+1]
			t2_sub_idx = data['subject'][fam_annotations[0]+idx+2]
			_key = str(t0_con)+str(t1_con)+str(t2_con)+str(t0_sub_idx)+str(t1_sub_idx)+str(t2_sub_idx)
			trigram_hist[trigram_mapping_[_key]]+=1
			trigram_histogram_all_families[trigram_mapping_[_key]]+=1


		classification_data_trigram.append(trigram_hist)
	
	return trigram_histogram_all_families, classification_data_trigram, labels


if __name__ == '__main__':
	# sanity_check()
	# depressed_families = np.squeeze(fam_phq_group[np.where(fam_phq_group[:, -2]==1), 0].astype(np.int32))
	# depressed_families = np.squeeze(fam_phq_group[np.where(fam_phq_group[:, 2]>=10), 0].astype(np.int32))
	# depressed_families = get_depressed_families()
	families = np.unique(np.array(data['family']))

	# get families in lag
	if LAG_ANALYSIS:
		families_del_idx = []
		fam_phq_group_del_idx = []
		for fid, fam in enumerate(families):
			# pdb.set_trace()
			if fam not in lag_families[:, 0]:
				# np.delete(families, np.where(families==fam))
				# np.delete(fam_phq_group, np.where(fam_phq_group[:, 0]==str(fam)), axis=0)
				families_del_idx.append(fid)
		for fam in fam_phq_group[:, 0]:
			if int(fam) not in lag_families[:, 0]:
				fam_phq_group_del_idx.append(np.where(fam_phq_group[:, 0]==str(fam))[0])

		families = np.delete(families, families_del_idx)
		fam_phq_group = np.delete(fam_phq_group, fam_phq_group_del_idx, axis=0)

	# depressed_families = np.squeeze(fam_phq_group[np.where(fam_phq_group[:, -2]==1), 0].astype(np.int32))
	'''
	There are families without history of depression that have PHQ>10, so we remove them
	'''
	depressed_families = np.squeeze(fam_phq_group[np.intersect1d(np.where(fam_phq_group[:, 2]>=10), np.where(fam_phq_group[:, -2]==1)), 0].astype(np.int32))

	print('remaining families after lag consideration {1} depressed {1}/{2}'.format(LAG_LIMIT, len(depressed_families)
																						, len(families)))

	print(len(fam_phq_group))
	assert len(fam_phq_group) == len(families), 'Number of Families in csv and sds do not match for this criteria'
	# print('Families from sds file-{0} and csv-{1}'.format(len(families), len(fam_phq_group)))

	count = 0
	for fam in families:
		if str(fam) not in fam_phq_group[:, 0]:
			count+=1
			# print(fam)
	# print(count)

	bi_hist, bigram_data, bigram_labels = calculate_bigram_histogram(families, depressed_families)
	tri_hist, trigram_data, trigram_labels = calculate_trigram_histogram(families, depressed_families)
	# print(tri_hist)

	histogram = tri_hist
	mapping = trigram_mapping_


#  these stats are not sufficient as the constructs are skewed towards others and we are to consider the top k
	_stat_mat = np.array(histogram)
	_max = np.max(_stat_mat)

	_max_con = [k for k, v in mapping.items() if histogram[v] == _max]
	_min = np.min(_stat_mat)
	_min_con = [k for k, v in mapping.items() if histogram[v] == _min]
	_mean = np.mean(_stat_mat)
	_mean_con = [k for k, v in mapping.items() if histogram[v] == _mean]

	print('histogram stats max-{0} max_construct-{1} min-{2} min_construct-{3} mean-{4} mean_construct-{5}'
		.format(_max, _max_con, _min, _min_con, _mean, _mean_con))

	_sorted_dist = np.flip(np.sort(_stat_mat)) # flipped to descending order
	_sorted_con = []

	for ele in _sorted_dist:
		for construct_dynamics, hist_idx in mapping.items():
			if histogram[hist_idx] == ele:
				if construct_dynamics not in _sorted_con:
					_sorted_con.append(construct_dynamics)
	# print(_sorted_con)
	k = 5
	print('top {0} transitions {1} bottom {0} transistions {2}'.format(k, _sorted_con[:k], _sorted_con[-1*k:]))

	# with open('trigram_data.pkl', 'wb') as fp:
	# 	pickle.dump(classification_data_trigram, fp)
    #
	# with open('bigram_data.pkl', 'wb') as fp:
	# 	pickle.dump(classification_data_bigram, fp)

# TODO finish the classifier part
	basic_model = svm.SVC(probability=True,class_weight='balanced')
	model = Classifier(basic_model)
	model.normalize(bigram_data, bigram_labels, print_dist=True)
	model.split_data(nfolds=5)
	model.classify_and_predict(optimize_for='auc', grid_verbose=False, njobs=-1, folds=5)
	metrics = model.metrics()

	# below are the validation metrics
	best_idx = model.model.best_index_
	results = model.model.cv_results_
	conf_matrix = np.array([[results['mean_test_tn'][best_idx], results['mean_test_fp'][best_idx]], [results['mean_test_fn'][best_idx], results['mean_test_tp'][best_idx]]])
	# print('scores for bigrams kappa-{0:.3f}+-{1:.3f} acc-{2:.3f}+-{3:.3f} f1-{4:.3f}+-{5:.3f} auc-{6:.3f}+-{7:.3f} conf-{8}'.format(results['mean_test_kappa'][best_idx],
	# 	   results['std_test_kappa'][best_idx], results['mean_test_acc'][best_idx], results['std_test_acc'][best_idx], results['mean_test_f1'][best_idx],
	# 				   							results['std_test_f1'][best_idx], results['mean_test_auc'][best_idx], results['std_test_auc'][best_idx], conf_matrix))

	# these are test metrics
	print(metrics)
	# print('scores for bigrams kappa-{0:.3f} acc-{1:.3f} f1-{2:.3f} auc-{3:.3f} conf-{4}'.format(metrics['kappa'], metrics['accuracy'], metrics['f1'], metrics['auc'], metrics['confusion']))


	model = Classifier(basic_model)
	model.normalize(trigram_data, trigram_labels)
	model.split_data(nfolds=5)
	model.classify_and_predict(optimize_for='auc', grid_verbose=False, njobs=-1, folds=5)
	metrics = model.metrics()

	# below are the validation metrics
	best_idx = model.model.best_index_
	results = model.model.cv_results_
	conf_matrix = np.array([[results['mean_test_tn'][best_idx], results['mean_test_fp'][best_idx]], [results['mean_test_fn'][best_idx], results['mean_test_tp'][best_idx]]])
	# print('scores for trigrams kappa-{0:.3f}+-{1:.3f} acc-{2:.3f}+-{3:.3f} f1-{4:.3f}+-{5:.3f} auc-{6:.3f}+-{7:.3f} conf-{8}'.format(
	# 	results['mean_test_kappa'][best_idx],
	# 	results['std_test_kappa'][best_idx], results['mean_test_acc'][best_idx], results['std_test_acc'][best_idx],
	# 	results['mean_test_f1'][best_idx],
	# 	results['std_test_f1'][best_idx], results['mean_test_auc'][best_idx], results['std_test_auc'][best_idx], conf_matrix))

	# these are test metrics
	print(metrics)
	# print('scores for bigrams kappa-{0:.3f} acc-{1:.3f} f1-{2:.3f} auc-{3:.3f} conf-{4}'.format(metrics['kappa'], metrics['f1'], metrics['auc'], metrics['confusion']))
