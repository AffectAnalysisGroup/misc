import cv2
import os, csv

root_dir = '/run/user/1435715183/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT'
video_dir = os.path.join(root_dir, 'Video_Data/CameraA/converted')
stop_construct_path = os.path.join(root_dir, 'LIFE Coding Stop Frame Constructs/Reliability testing/TXT Files')

ann_files = os.listdir(stop_construct_path)
_header = [0, 82, 83, 84, 85]

vid_files_org = [file.split('_') for file in os.listdir(video_dir)]
vid_files = [file[0]+'_'+file[1] if len(file)>2 else None for file in vid_files_org]


for file in ann_files:
	_time = []
	filename = file.split('_')[:-1]
	_vid_file =  vid_files_org[vid_files.index(filename[0][2:]+'_'+filename[1])]
	vid_file = _vid_file[0]
	for ele in _vid_file[1:]:
		vid_file = vid_file+'_'+ele
	if file.endswith('CM.txt'):
		csv_fp = open(filename[0]+'_'+filename[1] + '_CB45_BO_CM.csv', 'w')
		print(filename[0]+'_'+filename[1] + '_CB45_BO_CM.csv')
	else:
		csv_fp = open(filename[0] + '_' + filename[1] + '_DB45_BO_KH.csv', 'w')
		print(filename[0] + '_' + filename[1] + '_DB45_BO_KH.csv')
	csvwriter = csv.writer(csv_fp)
	csvwriter.writerow(_header)
	# with open(os.path.join(video_dir, vid_file), 'r') as fp:
	vcap = cv2.VideoCapture()
	_ret = vcap.open(os.path.join(video_dir, vid_file))
	if _ret:
		_fps = vcap.get(cv2.CAP_PROP_FPS)
		_num_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
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
	print(file, _num_frames)
	# break