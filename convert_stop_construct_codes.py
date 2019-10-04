import cv2
import os, csv, pdb
import numpy as np
from numpy import genfromtxt
# import matplotlib.plyplot as plt

root_dir = '/run/user/1435715183/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT'
video_dir = os.path.join(root_dir, 'Video_Data/CameraA/converted')
stop_construct_path = os.path.join(root_dir, 'LIFE', 'LIFE Coding Stop Frame Constructs/Reliability testing/TXT Files')
ind_csv_path = os.path.abspath('independent_annotations')
win_csv_path = os.path.abspath('windowed_annotations')

# TODO fix this for video convenience
ann_files = os.listdir(stop_construct_path)[8:-3]
# print(ann_files)
_header = [0, 82, 83, 84, 85]

vid_files_org = [file.split('_') for file in os.listdir(video_dir)]
vid_files = [file[0]+'_'+file[1] if len(file)>2 else None for file in vid_files_org]
root_name = []
frame_rates = []
nframes = []
agreement = True

# TODO remove these
# nframes = [50]
# frame_rates = [2]

def closest_prev_construct(frame, rating_pos, ref_array, nsec_frames):


	# print('closest')
	_diff = frame-rating_pos
	win_diff =_diff[np.where(abs(_diff)<=nsec_frames)[0]]
	_min = 1e10

	for d in win_diff:
		if _min>d >= 0: #>=0 because it should be a prev. annotation
			_min = d

	if _min == 1e10:
		prev_construct = 3
	else:
		prev_construct = np.where(ref_array[rating_pos[np.where(_diff ==_min)], 1:]==1)[1]

	return prev_construct

def generate_csvs(ann_files):

	for file in ann_files[0:12]:
		_time = []
		filename = file.split('_')[:-1]
		root_name.append(filename[0]+'_'+filename[1])
		_vid_file =  vid_files_org[vid_files.index(filename[0][2:]+'_'+filename[1])]
		vid_file = _vid_file[0]
		for ele in _vid_file[1:]:
			vid_file = vid_file+'_'+ele
		if file.endswith('CM.txt'):
			csv_fp = open(os.path.join(ind_csv_path, filename[0]+'_'+filename[1] + '_CB45_BO_CM.csv'), 'w')
			# print(filename[0]+'_'+filename[1] + '_CB45_BO_CM.csv')
		else:
			csv_fp = open(os.path.join(ind_csv_path, filename[0] + '_' + filename[1] + '_DB45_BO_KH.csv'), 'w')
			# print(filename[0] + '_' + filename[1] + '_DB45_BO_KH.csv')
		csvwriter = csv.writer(csv_fp)
		csvwriter.writerow(_header)
		# with open(os.path.join(video_dir, vid_file), 'r') as fp:
		vcap = cv2.VideoCapture()
		_ret = vcap.open(os.path.join(video_dir, vid_file))
		if _ret:
			vcap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
			_dur = vcap.get(cv2.CAP_PROP_POS_MSEC)

			_fps = vcap.get(cv2.CAP_PROP_FPS)
			frame_rates.append(_fps)
			_num_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
			# _num_frames = _fps*_dur*0.001#vcap.get(cv2.CAP_PROP_FRAME_COUNT)
			nframes.append(_num_frames)
		_data = []
		with open(os.path.join(stop_construct_path, file), 'r') as fp:
			for line in fp:
				_data.append(line.split('\t'))
			# _time.append(_data[-1][-2]*_fps)
		_data.sort(key=lambda _sample:_sample[-2])
		data = [[int(float(timestamp)*_fps), 1, 0, 0, 0] if construct.lower() == 'aggressive' else [int(float(timestamp)*_fps), 0, 1, 0, 0]
		if construct.lower() == 'dysphoric' else [int(float(timestamp)*_fps), 0, 0, 1, 0]
		if construct.lower() == 'positive' else [int(float(timestamp)*_fps), 0, 0, 0, 1]
		for (construct, _, minute_time, timestamp, _) in _data]
		#
		idx = 0
		prev = [0, 0, 0, 0, 0]
		for row in data:
			if row[0] == idx:
				csvwriter.writerow(row)
			else:
				for i in range(1, row[0]-idx):
					# csvwriter.writerow([idx+i, prev[1], prev[2], prev[3], prev[4]]) # frame-level annotation
					csvwriter.writerow([idx+i, 0, 0, 0, 0]) # only onset annotation
				csvwriter.writerow(row)
			idx = row[0]
			prev = row
		for i in range(1, int(_num_frames)-row[0]+1):
			# csvwriter.writerow([row[0]+i, row[1], row[2], row[3], row[4]]) # frame-level annotation
			csvwriter.writerow([row[0]+i, 0, 0, 0, 0]) # only onset annotation
		fp.close()
		# print(file, _num_frames)
	# break

	return
# 58421_01_

def generate_csvs_windowed(nsecs):

	print('comparison')
	disagreement_prop = [[], []]
	conf_matrix = np.zeros((2, 4,4))

	for fid, filename in enumerate(root_name):
		print('-'*20)
		print(filename)
		print('-'*20)

		nsec_frames = frame_rates[fid] * nsecs

		csv_file1 = os.path.join(ind_csv_path, filename + '_CB45_BO_CM.csv')
		csv_file2 = os.path.join(ind_csv_path, filename+'_DB45_BO_KH.csv')

		# csv_file1 = './sample1.csv'
		# csv_file2 = './sample_w9.csv'

		ann1 = genfromtxt(csv_file1, delimiter=',', skip_header=True).astype(np.int32)
		ann2 = genfromtxt(csv_file2, delimiter=',', skip_header=True).astype(np.int32)
		# ann2 = np.zeros(ann1.shape)

		rating_pos1 = np.where(np.sum(ann1[:, 1:], axis=1)>0)[0] # get annotated frame indices
		rating_pos2 = np.where(np.sum(ann2[:, 1:], axis=1)>0)[0]

		del_frame1 = []
		del_frame2 = []

		for frame in rating_pos1:
			# if frame == rating_pos1[-2]:
				# print('here')
			_diff = np.abs(rating_pos2 - frame) # normalize the ref. frame numbers with the current frame number
			_construct = np.where(ann1[frame, 1:]==1) # get the current frame's construct
			if len(np.where(_diff<=nsec_frames)[0]) == 0: # if no frames annotated by the other in the window
				del_frame1.append(frame)
				_prev_construct = 3
				conf_matrix[0, _construct, _prev_construct] += 1
			else: # if there are annotated frames from the window
				if agreement:
					win_frames = np.where(_diff <= nsec_frames) # get annotated frames
					# conf_matrix[0, _construct, np.where(ann2[rating_pos2[win_frames], 1:]==1)[1]] += 1
					if np.sum(ann2[rating_pos2[win_frames], 1:], axis=0)[_construct] < 1: # check if those frames have atleast one annotations corresponding to the construct
						del_frame1.append(frame)
						_prev_construct = closest_prev_construct(frame, rating_pos2, ann2, nsec_frames) #change this to handle multiple annotations in the window
						conf_matrix[0, _construct, _prev_construct] += 1
					else:
						_prev_construct = _construct
						conf_matrix[0, _construct, _prev_construct] += 1

			# print(_construct, _prev_construct)

		# print('deleting {0} annotated frames, total-{1} annotated frames'.format(len(del_frame1), len(rating_pos1)))

		for frame in rating_pos2:
			_diff = np.abs(rating_pos1 - frame)  # normalize the ref. frame numbers with the current frame number
			_construct = np.where(ann2[frame, 1:] == 1)  # get the current frame's construct
			if len(np.where(_diff <= nsec_frames)[0]) == 0:  # if no frames annotated by the other in the window
				del_frame2.append(frame)
				conf_matrix[1, _construct, -1] += 1
			else:  # if there are annotated frames from the window
				if agreement:
					win_frames = np.where(_diff <= nsec_frames)  # get annotated frames
					# conf_matrix[0, _construct, np.where(ann2[rating_pos2[win_frames], 1:]==1)[1]] += 1
					if np.sum(ann1[rating_pos1[win_frames], 1:], axis=0)[
						_construct] < 1:  # check if those frames have atleast one annotations corresponding to the construct
						del_frame1.append(frame)
						_prev_construct = closest_prev_construct(frame, rating_pos1,
																 ann1, nsec_frames)  # change this to handle multiple annotations in the window
						conf_matrix[1, _construct, _prev_construct] += 1
					else:
						conf_matrix[1, _construct, _construct] += 1

		# for frame in rating_pos2:
		# 	_diff = np.abs(rating_pos1 - frame)
		# 	_construct = np.where(ann2[frame, 1:]==1)
		# 	if len(np.where(_diff<nsec_frames)[0]) == 0:
		# 		del_frame2.append(frame)
		# 	else:
		# 		if agreement:
		# 			win_frames = np.where(_diff < nsec_frames)
		# 			conf_matrix[1, _construct, np.where(ann1[rating_pos1[win_frames], 1:]==1)[1]] += 1
		# 			if np.sum(ann1[rating_pos1[win_frames], 1:], axis=0)[_construct] < 1:
		# 				del_frame2.append(frame)
		# # print('deleting {0} annotated frames, total-{1} annotated frames'.format(len(del_frame2), len(rating_pos2)))

		ann1[del_frame1, 1:] = [0, 0, 0, 0]
		ann2[del_frame2, 1:] = [0, 0, 0, 0]

		disagreement_prop[0].append(len(del_frame1)/len(rating_pos1)) # disagreement of CM with KH
		disagreement_prop[1].append(len(del_frame2)/len(rating_pos2)) # disagreement of KH with CM

		print('agreement of CM with KH-{0:.4f}'.format(1-disagreement_prop[0][fid]))
		print('agreement of KH with CM-{0:.4f}'.format(1-disagreement_prop[1][fid]))


		# conf_matrix[0, ...] = np.divide(conf_matrix[0, ...], np.sum(conf_matrix[0, ...], axis=1).T+1e-8)
		# conf_matrix[1, ...] = np.divide(conf_matrix[1, ...], np.sum(conf_matrix[1, ...], axis=1).T+1e-8)

		# print(del_frame1)
		# print(del_frame2)

		# with open(os.path.join(win_csv_path, 'agreement/2sec', filename + '_CB45_BO_CM.csv'), 'w') as fp:
		# 	csvwriter = csv.writer(fp)
		# 	csvwriter.writerow(_header)
		# 	for row in ann1:
		# 		csvwriter.writerow([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])])
        #
		# with open(os.path.join(win_csv_path, 'agreement/2sec', filename + '_DB45_BO_KH.csv'), 'w') as fp:
		# 	csvwriter = csv.writer(fp)
		# 	csvwriter.writerow(_header)
		# 	for row in ann2:
		# 		csvwriter.writerow([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])])

	print('confusion matrix \n {0}'.format(conf_matrix.astype(np.float16)))
		# break

	return disagreement_prop

if __name__ == '__main__':
	generate_csvs(ann_files)
	# root_name = ['123456_01']
	root_name = list(dict.fromkeys(root_name))
	win_len = 1

	hist = []
	wins = []
	bar_width = 0.35
	# plt.subplots()

	# generate_csvs_windowed(1)

	for wid, win_len in enumerate(np.arange(1, 3, 0.5)):
		print('-' * 20)
		print('window duration-{0:.4f} sec'.format(win_len))
		print('-' * 20)
		hist.append(generate_csvs_windowed(win_len))
		wins.append(win_len)

	# 	plt.bar(np.arange(len(wins)), hist[wid][0], color='b', label='CM wrt KH')
	# 	plt.bar(np.arange(len(wins))+bar_width, hist[wid][1], color='b', label='KH wrt CM')
    #
	# plt.legend()
	# plt.show()


