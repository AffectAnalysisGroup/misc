import cv2
import os, csv
import numpy as np
from numpy import genfromtxt

root_dir = '/run/user/1435715183/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT'
video_dir = os.path.join(root_dir, 'Video_Data/CameraA/converted')
stop_construct_path = os.path.join(root_dir, 'LIFE Coding Stop Frame Constructs/Reliability testing/TXT Files')
ind_csv_path = os.path.abspath('independent_annotations')
win_csv_path = os.path.abspath('windowed_annotations')

ann_files = os.listdir(stop_construct_path)
_header = [0, 82, 83, 84, 85]

vid_files_org = [file.split('_') for file in os.listdir(video_dir)]
vid_files = [file[0]+'_'+file[1] if len(file)>2 else None for file in vid_files_org]
root_name = []
frame_rates = []
nframes = []
agreement = False

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
	for fid, filename in enumerate(root_name):
		print(filename)
		nsec_frames = frame_rates[fid] * nsecs

		csv_file1 = os.path.join(ind_csv_path, filename + '_CB45_BO_CM.csv')
		csv_file2 = os.path.join(ind_csv_path, filename+'_DB45_BO_KH.csv')

		ann1 = genfromtxt(csv_file1, delimiter=',', skip_header=True).astype(np.int32)
		ann2 = genfromtxt(csv_file2, delimiter=',', skip_header=True).astype(np.int32)
		# ann2 = np.zeros(ann1.shape)

		rating_pos1 = np.where(np.sum(ann1[:, 1:], axis=1)>0)[0]
		rating_pos2 = np.where(np.sum(ann2[:, 1:], axis=1)>0)[0]

		del_frame1 = []
		del_frame2 = []
		for fid, frame in enumerate(rating_pos1):
			_diff = np.abs(rating_pos2 - frame)
			_construct = np.where(ann1[frame, 1:]==1)
			if len(np.where(_diff<nsec_frames)[0]) == 0:
				del_frame1.append(frame)
			else:
				if agreement:
					win_frames = np.where(_diff < nsec_frames)
					if np.sum(ann2[rating_pos2[win_frames]], axis=0)[_construct] < 1:
						del_frame1.append(frame)

		print('deleting {0} annotated frames, total-{1} annotated frames'.format(len(del_frame1), len(rating_pos1)))

		for fid, frame in enumerate(rating_pos2):
			_diff = np.abs(rating_pos1 - frame)
			_construct = np.where(ann2[frame, 1:]==1)
			if len(np.where(_diff<nsec_frames)[0]) == 0:
				# if agreement:
				# 	win_frames = np.where(_diff<nsec_frames)
				# 	if np.sum(ann2[win_frames], axis=0)[_construct] < 1:
				del_frame2.append(frame)
			else:
				if agreement:
					win_frames = np.where(_diff < nsec_frames)
					if np.sum(ann1[rating_pos1[win_frames]], axis=0)[_construct] < 1:
						del_frame2.append(frame)
		print('deleting {0} annotated frames, total-{1} annotated frames'.format(len(del_frame2), len(rating_pos2)))

		ann1[del_frame1, 1:] = [0, 0, 0, 0]
		ann2[del_frame2, 1:] = [0, 0, 0, 0]

		with open(os.path.join(win_csv_path, filename + '_CB45_BO_CM.csv'), 'w') as fp:
			csvwriter = csv.writer(fp)
			csvwriter.writerow(_header)
			for row in ann1:
				csvwriter.writerow([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])])

		with open(os.path.join(win_csv_path, filename + '_DB45_BO_KH.csv'), 'w') as fp:
			csvwriter = csv.writer(fp)
			csvwriter.writerow(_header)
			for row in ann2:
				csvwriter.writerow([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])])


		# np.savetxt(os.path.join(win_csv_path, filename + '_CB45_BO_CM.csv'), ann1.astype(int), delimiter=',', header = '0, 80, 81,82,83')
		# np.savetxt(os.path.join(win_csv_path, filename+'_DB45_BO_KH.csv'), ann2.astype(int), delimiter=',', header = '0, 80, 81,82,83')

		# break

	return

if __name__ == '__main__':
	generate_csvs(ann_files)
	root_name = list(dict.fromkeys(root_name))
	generate_csvs_windowed(5)


