import csv, os, pdb
import cv2
import numpy as np

# server_dir = '/run/user/1435715183/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/'

server_dir = '/run/user/1435715183/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/BP4D++/FACS/OCC/ELAN Projects/Exported'
video_dir = '/run/user/1435715183/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=rawdata/BP4D++/Video_Data/mjpeg_FACS only'
out_dir = './BP4D++_occ_reliability/'

frame_level_info = []
# _fps = 30
# _dur_ = 10

# list of indices of the AUs. 0 is for reliability calculation in (0, 0)
allowed_aus = list(range(0, 21))+[22, 23, 24]+list(range(27, 40))+[99]
allowed_aus[0] = 1

for _csv in os.listdir(server_dir):
    if _csv.endswith('.csv') and 'Test' not in _csv:
        vid_file1 = _csv.split('_')[0]+'_'+_csv.split('_')[1]+'.avi'
        vid_file2 = _csv.split('_')[0] + '_' + _csv.split('_')[1] +'-MJPG'+'.avi'
        with open(os.path.join(server_dir, _csv), 'r') as fp:

            csvreader = csv.reader(fp)
            vcap = cv2.VideoCapture()
            if vid_file1 in os.listdir(video_dir):
                vid_file = vid_file1
            elif vid_file2 in os.listdir(video_dir):
                vid_file = vid_file2
            else:
                raise FileNotFoundError

            _ret = vcap.open(os.path.join(video_dir, vid_file))

            if _ret:
                vcap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                _dur_ = vcap.get(cv2.CAP_PROP_POS_MSEC) // 1e3
                _fps = vcap.get(cv2.CAP_PROP_FPS)

                if _fps > 60: # A relatively high frequency as reference to prevent absurd fps values
                    print('Video fps corruption')
                    raise ValueError

            print(_dur_, _fps)
            out_csv = np.zeros((int(_dur_*_fps), len(allowed_aus)-1)) # ignore the header and frame no. column
            out_csv = np.append(np.reshape(range(1, int(_dur_*_fps)+1), (-1, 1)), out_csv, axis=1)

            for line in csvreader:
                if 'Coding' not in line:
                    AU, _start, _stop  = (int(line[0].split(' ')[1]), float(line[2]), float(line[3]))
                    out_csv[int(_start*_fps):int(_stop*_fps), allowed_aus.index(AU)] = 1

        # out_csv[0, 0] = 1 # setting the header (0, 0) to 1 for reliability software
        with open(os.path.join(out_dir, _csv), 'w') as fp:
            csvwriter = csv.writer(fp)
            csvwriter.writerow(allowed_aus)

            for line in out_csv:
                csvwriter.writerow(line)