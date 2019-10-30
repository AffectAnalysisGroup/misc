from datetime import datetime
import cv2
import os, csv, pdb
import numpy as np
from numpy import genfromtxt
from reliability_emo_onset import calculate_kappa, cal_2afc

# import matplotlib.plyplot as plt

constructs = ['Aggressive', 'Dysphoric', 'Positive', 'Other']

root_dir = '/run/user/1435715183/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT'
video_dir = os.path.join(root_dir, 'Video_Data/CameraA/converted')
# stop_construct_path = os.path.join(root_dir, 'LIFE', 'LIFE Coding Stop Frame Constructs/Reliability testing/TXT Files')
stop_construct_path = os.path.join(root_dir, 'LIFE', 'LIFE Coding Stop Frame Constructs/PSI Task/TXT Files')
ind_csv_path = os.path.abspath('independent_annotations')
win_csv_path = os.path.abspath('windowed_annotations')

# TODO fix this for video convenience
# ann_files = os.listdir(stop_construct_path)#[8:-3]

# TODO uncomment for weekly analysis
_ann_files = os.listdir(stop_construct_path)#[8:-3]
week_files = ['1020862', '1007091', '1007092', '1028622']
ann_files = [x if x.split('_')[0] in week_files else None for x in _ann_files]
ann_files = list(filter(lambda x:x!=None, ann_files))
fp_weekly = open('TPOT_weekly_report_'+datetime.today().strftime('%Y_%m_%d')+'.csv', 'w')
weekly_csvwriter = csv.writer(fp_weekly)

_header = [0, 82, 83, 84, 85]

vid_files_org = [file.split('_') for file in os.listdir(video_dir)]
vid_files = [file[0] + '_' + file[1] if len(file) > 2 else None for file in vid_files_org]
root_name = []
frame_rates = []
nframes = []
agreement = True


def closest_prev_construct(frame, rating_pos, ref_array, nsec_frames):
    _diff = frame - rating_pos
    win_diff = _diff[np.where(abs(_diff) <= nsec_frames)[0]]
    _min = 1e10

    for d in win_diff:
        if _min > d >= 0:  # >=0 because it should be a prev. annotation
            _min = d

    if _min == 1e10:
        prev_construct = 3
    else:
        prev_construct = np.where(ref_array[rating_pos[np.where(_diff == _min)], 1:] == 1)[1]

    return prev_construct


def compare_windows(curr_rating_pos, ref_rating_pos, curr_ann, ref_ann, conf_matrix, nsec_frames):
    del_frame = []

    for frame in curr_rating_pos:
        _diff = np.abs(ref_rating_pos - frame)  # normalize the ref. frame numbers with the current frame number
        _construct = np.where(curr_ann[frame, 1:] == 1)  # get the current frame's construct
        if len(np.where(_diff <= nsec_frames)[0]) == 0:  # if no frames annotated by the other in the window
            del_frame.append(frame)
            conf_matrix[_construct, -1] += 1
        else:  # if there are annotated frames from the window
            if agreement:
                win_frames = np.where(_diff <= nsec_frames)  # get annotated frames
                # conf_matrix[0, _construct, np.where(ann2[rating_pos2[win_frames], 1:]==1)[1]] += 1
                if np.sum(ref_ann[ref_rating_pos[win_frames], 1:], axis=0)[
                    _construct] < 1:  # check if those frames have atleast one annotations corresponding to the construct
                    del_frame.append(frame)
                    _prev_construct = closest_prev_construct(frame, ref_rating_pos,
                                                             ref_ann,
                                                             nsec_frames)  # change this to handle multiple annotations in the window
                    conf_matrix[_construct, _prev_construct] += 1
                else:
                    conf_matrix[_construct, _construct] += 1

    return conf_matrix, del_frame


def generate_csvs(ann_files):

    for file in ann_files:

        filename = file.split('_')[:-1]
        root_name.append(filename[0] + '_' + filename[1])
        _vid_file = vid_files_org[vid_files.index(filename[0][2:] + '_' + filename[1])]
        vid_file = _vid_file[0]

        for ele in _vid_file[1:]:
            vid_file = vid_file + '_' + ele
        if file.endswith('CM.txt'):
            csv_fp = open(os.path.join(ind_csv_path, filename[0] + '_' + filename[1] + '_CM.csv'), 'w')
        else:
            csv_fp = open(os.path.join(ind_csv_path, filename[0] + '_' + filename[1] + '_KH.csv'), 'w')

        csvwriter = csv.writer(csv_fp)
        csvwriter.writerow(_header)
        vcap = cv2.VideoCapture()
        _ret = vcap.open(os.path.join(video_dir, vid_file))

        if _ret:
            vcap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            _dur = vcap.get(cv2.CAP_PROP_POS_MSEC)/1e3

            _fps = vcap.get(cv2.CAP_PROP_FPS)
            if _fps > 60:
                _fps = 29.97 # default value
            frame_rates.append(_fps)
            _num_frames  = _fps * _dur
            # _num_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
            # _num_frames = _fps*_dur*0.001#vcap.get(cv2.CAP_PROP_FRAME_COUNT)
            nframes.append(_num_frames)

        _data = []

        with open(os.path.join(stop_construct_path, file), 'r') as fp:
            for line in fp:
                _data.append(line.split('\t'))
        _data.sort(key=lambda _sample: _sample[-2])
        data = [[int(float(timestamp) * _fps), 1, 0, 0, 0] if construct.lower() == 'aggressive' else [
            int(float(timestamp) * _fps), 0, 1, 0, 0]
        if construct.lower() == 'dysphoric' else [int(float(timestamp) * _fps), 0, 0, 1, 0]
        if construct.lower() == 'positive' else [int(float(timestamp) * _fps), 0, 0, 0, 1]
                for (construct, _, minute_time, timestamp, _) in _data]
        #
        idx = 0
        prev = [0, 0, 0, 0, 0]
        for row in data:
            if row[0] == idx:
                csvwriter.writerow(row)
            else:
                for i in range(1, row[0] - idx):
                    # csvwriter.writerow([idx+i, prev[1], prev[2], prev[3], prev[4]]) # frame-level annotation
                    csvwriter.writerow([idx + i, 0, 0, 0, 0])  # only onset annotation
                csvwriter.writerow(row)
            idx = row[0]
            prev = row
        for i in range(1, int(_num_frames) - row[0] + 1):
            # csvwriter.writerow([row[0]+i, row[1], row[2], row[3], row[4]]) # frame-level annotation
            csvwriter.writerow([row[0] + i, 0, 0, 0, 0])  # only onset annotation
        fp.close()

    return

def generate_csvs_windowed(nsecs):
    conf_matrix = np.zeros((2, 2, 4, 4)) # parent/child x annotator A/B x A's construct x B's construct
    disagreement_prop = [[], []]

    for fid, filename in enumerate(root_name):

        nsec_frames = frame_rates[fid] * nsecs

        # csv_file1 = os.path.join(ind_csv_path, filename + '_CB45_BO_CM.csv')
        # csv_file2 = os.path.join(ind_csv_path, filename + '_DB45_BO_KH.csv')

        csv_file1 = os.path.join(ind_csv_path, filename + '_CM.csv')
        csv_file2 = os.path.join(ind_csv_path, filename + '_KH.csv')


        ann1 = genfromtxt(csv_file1, delimiter=',', skip_header=True).astype(np.int32)
        ann2 = genfromtxt(csv_file2, delimiter=',', skip_header=True).astype(np.int32)

        rating_pos1 = np.where(np.sum(ann1[:, 1:], axis=1) > 0)[0]  # get annotated frame indices
        rating_pos2 = np.where(np.sum(ann2[:, 1:], axis=1) > 0)[0]

        if filename.split('_')[0][-1] == '1':
        #     Child
            char = 0
        else:
            char = 1

        conf_matrix[char, 0, :, :], del_frame1 = compare_windows(rating_pos1, rating_pos2, ann1, ann2, conf_matrix[char, 0, :, :],
                                                           nsec_frames)
        conf_matrix[char, 1, :, :], del_frame2 = compare_windows(rating_pos2, rating_pos1, ann2, ann1, conf_matrix[char, 1, :, :],
                                                           nsec_frames)

        ann1[del_frame1, 1:] = [0, 0, 0, 0]
        ann2[del_frame2, 1:] = [0, 0, 0, 0]

        disagreement_prop[0].append(len(del_frame1) / len(rating_pos1))  # disagreement of CM with KH
        disagreement_prop[1].append(len(del_frame2) / len(rating_pos2))  # disagreement of KH with CM

        # print('agreement of CM with KH-{0:.4f}'.format(1 - disagreement_prop[0][fid]))
        # print('agreement of KH with CM-{0:.4f}'.format(1 - disagreement_prop[1][fid]))

        if 'annotator1_all' not in locals():
            annotator1_all = ann1
        else:
            annotator1_all = np.append(annotator1_all, ann1, axis=0)

        if 'annotator2_all' not in locals():
            annotator2_all = ann2
        else:
            annotator2_all = np.append(annotator2_all, ann2, axis=0)

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

    # print('confusion matrix \n {0}'.format(conf_matrix.astype(np.float16)))
    # print('confusion matrix \n {0}'.format(conf_matrix2.astype(np.float16)))
    # break
    kappa = np.zeros((conf_matrix.shape[0], conf_matrix.shape[1], 4))

    for char in range(conf_matrix.shape[0]): # child or parent
        for a in range(conf_matrix.shape[1]):
            kappa[char, a, :] = calculate_kappa(conf_matrix[char, a, ...])
        kappa[char, ...] = np.reshape(kappa[char, ...], (-1, 4))

        if char == 0:
            np.savez('./graph_data/tpot/tpot_Child_kappa' + str(float(nsecs)), kappa[0, ...])
        else:
            np.savez('./graph_data/tpot/tpot_Parent_kappa' + str(float(nsecs)), kappa[1, ...])
    # break
    # print('{0} {1}sec conf_matrix-{2} \n '.format(char, WINDOW, conf_matrix))
    # print(kappa)

    # afc_score = cal_2afc(annotator1_all, annotator2_all)
    # print('afc score', afc_score)
    #
    # del annotator1_all, annotator2_all

    kappa_stats = np.around(np.concatenate((np.nanmean(kappa, axis=0).T, np.nanmin(kappa, axis=0).T, np.nanmax(kappa, axis=0).T,
                  np.nanstd(kappa, axis=0).T), axis=1), 3) # rounding to 3 decimal places

    weekly_csvwriter.writerow(['Child']+5*['\t']+['Parent'])
    weekly_csvwriter.writerow(['Construct', 'Mean', 'Min', 'Max', 'Std', str(nsecs)+' secs', 'Mean', 'Min', 'Max', 'Std'])

    for affect in range(kappa_stats.shape[0]):
        _row = [constructs[affect]]+[kappa_stats[affect, 2*i] for i in range(0, 4)]+['\t']+[kappa_stats[affect, 2*i+1] for i in range(0, 4)]
        weekly_csvwriter.writerow(_row) # even column indices are child
    # odd column indices are parent

    print('kappa stats \n Mean-{0}\n Min-{1}\n Max-{2}\n Std-{3}\n'.format(np.nanmean(kappa, axis=0), np.nanmin(kappa, axis=0), np.nanmax(kappa, axis=0),
                  np.nanstd(kappa, axis=0)))

    return disagreement_prop


if __name__ == '__main__':
    generate_csvs(ann_files)
    root_name = list(dict.fromkeys(root_name))
    win_len = 1


    for wid, win_len in enumerate(np.arange(0.5, 3, 0.5)):
        kappa = np.zeros((2,2,4))
        weekly_csvwriter.writerow(['\n'])
        generate_csvs_windowed(win_len)

    weekly_csvwriter.writerow(['* Cohen\'s Kappa'])
    fp_weekly.close()
