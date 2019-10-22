import os, pdb
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import scipy.io as sio

# from convert_stop_construct_codes import closest_prev_construct, compare_windows

LIFE_path = '/run/user/1435715183/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=groundtruthdata/EMO/LIFE/' \
            '2018.11.27_Converted to Lab Standard/Construct_Onset'

video_path = '/run/user/1435715183/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/Emotion/Video_Data'
mapping_file = 'DNT-Emotion_NamingMappingTable-LIFE.xlsx'

mapping_annotation_video = \
    pd.read_excel(mapping_file, skiprows=1, skip_footer=2, index_col=None).set_index('Original filename')[
        'New Filename'].to_dict()
# print(mapping_annotation_video)#'['Original filename'][0])

DYAD = True
TRIAD = False
WINDOW = 1
agreement = True

annotators = {'CH': 0, 'CO': 1, 'MZ': 2, 'MN': 3}
conf_matrix = np.zeros((4, 4, 4, 4), dtype=np.int64)
score = np.zeros((2, 4))  # constructs x annotators x annotators
ann = np.array((1, 4))

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


def compare_windows(file, curr_rating_pos, ref_rating_pos, curr_ann, ref_ann, conf_matrix, nsec_frames):
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

                # if file.startswith('2143212'):
                #     print('here')

                if len(np.sum(ref_ann[ref_rating_pos[win_frames], 1:], axis=0)[
                    _construct]) < 2 and np.sum(ref_ann[ref_rating_pos[win_frames], 1:], axis=0)[
                    _construct] < 1:  # check if those frames have atleast one annotations corresponding to the construct
                    del_frame.append(frame)
                    _prev_construct = closest_prev_construct(frame, ref_rating_pos,
                                                             ref_ann,
                                                             nsec_frames)  # change this to handle multiple annotations in the window
                    conf_matrix[_construct, _prev_construct] += 1
                else:
                    conf_matrix[_construct, _construct] += 1

    return conf_matrix, del_frame


def get_video_stats(filename):
    frame_rates = []
    nframes = []
    _filename = filename.split('_')
    # video_name = mapping_annotation_video[_filename[0][:-1]+_filename[-1].split('.')[0]+'.dat'].split('_')
    # video_files = os.path.join(video_path, video_name[0]+'_'+video_name[1])

    # for vid_file in video_files:
    # vcap = cv2.VideoCapture()
    # _ret = vcap.open(video_file)
    # if _ret:
    #     vcap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    #     _dur = vcap.get(cv2.CAP_PROP_POS_MSEC)
    #
    #     _fps = vcap.get(cv2.CAP_PROP_FPS)
    #     frame_rates.append(_fps)
    #     _num_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    #     # _num_frames = _fps*_dur*0.001#vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    #     nframes.append(_num_frames)
    #     print(_filename, _num_frames, _fps)

    for file in filename:
        frame_rates.append(29.97)

    return frame_rates


def calculate_conf_matrix(csv_path, filename1, filename2, frame_rate, annotator1_id, annotator2_id, nsecs):
    disagreement_prop = [[], []]

    # for fid, filename in enumerate(root_name):


    nsec_frames = frame_rate * nsecs

    csv_file1 = os.path.join(os.path.join(LIFE_path, csv_path), filename1)
    csv_file2 = os.path.join(os.path.join(LIFE_path, csv_path), filename2)

    # csv_file1 = './sample1.csv'
    # csv_file2 = './sample_w9.csv'

    ann1 = np.genfromtxt(csv_file1, delimiter=',', skip_header=True).astype(np.int32)
    ann2 = np.genfromtxt(csv_file2, delimiter=',', skip_header=True).astype(np.int32)
    # ann2 = np.zeros(ann1.shape)

    rating_pos1 = np.where(np.sum(ann1[:, 1:], axis=1) > 0)[0]  # get annotated frame indices
    rating_pos2 = np.where(np.sum(ann2[:, 1:], axis=1) > 0)[0]

    conf_matrix[annotator1_id, annotator2_id, :, :], del_frame1 = compare_windows(file, rating_pos1, rating_pos2, ann1, ann2, conf_matrix[annotator1_id, annotator2_id, :, :],
                                                       nsec_frames)
    conf_matrix[annotator2_id, annotator1_id, :, :], del_frame2 = compare_windows(file, rating_pos2, rating_pos1, ann2, ann1, conf_matrix[annotator2_id, annotator1_id, :, :],
                                                       nsec_frames)

    ann1[del_frame1, 1:] = [0, 0, 0, 0]
    ann2[del_frame2, 1:] = [0, 0, 0, 0]

    # assert ann1.shape == ann2.shape # seems like some error in the EMO csv files

    max_frames = min(ann1.shape[0], ann2.shape[0]) # due to the above issue in the csvs
    ann1 = ann1[:max_frames, ...]
    ann2 = ann2[:max_frames, ...]

    disagreement_prop[0].append(len(del_frame1) / len(rating_pos1))  # disagreement of CM with KH
    disagreement_prop[1].append(len(del_frame2) / len(rating_pos2))  # disagreement of KH with CM

    # print('agreement of {0} with {1}-{2:.4f}'.format(annotator1_id, annotator2_id, 1 - disagreement_prop[0][-1]))
    # print('agreement of {0} with {1}-{2:.4f}'.format(annotator2_id, annotator1_id, 1 - disagreement_prop[1][-1]))


    # print('confusion matrix \n {0} \n {1}'.format(conf_matrix[annotator1_id, annotator2_id].astype(np.int64), conf_matrix[annotator2_id, annotator1_id].astype(np.int64)))

    return disagreement_prop, ann1, ann2

def calculate_kappa(conf_matrix):

    # kappa here is Cohen's kappa
    # implementation reference:
    # https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
    # kappa range= [-1, 1]; negative kappa indicates disagreement
    # more about kappa-https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/

    kappa = []

    for construct in range(4):
        binarized_confusion_matrix = np.zeros((2, 2))

        binarized_confusion_matrix[0, 0] = conf_matrix[construct, construct]
        binarized_confusion_matrix[0, 1] = np.sum(conf_matrix[construct, :]) - conf_matrix[construct, construct]
        binarized_confusion_matrix[1, 0] = np.sum(conf_matrix[:, construct]) - conf_matrix[construct, construct]
        binarized_confusion_matrix[1, 1] = np.sum(conf_matrix) - np.sum(binarized_confusion_matrix)

        observed_acc = (binarized_confusion_matrix[0, 0]+binarized_confusion_matrix[1,1])/np.sum(binarized_confusion_matrix)
        expected_acc = np.sum(binarized_confusion_matrix[0, :]) * np.sum(binarized_confusion_matrix[:, 0])/np.sum(binarized_confusion_matrix)
        expected_acc += np.sum(binarized_confusion_matrix[1, :]) * np.sum(binarized_confusion_matrix[:, 1])/np.sum(binarized_confusion_matrix)
        expected_acc = expected_acc/np.sum(binarized_confusion_matrix)

        k = (observed_acc - expected_acc)/(1 - expected_acc)
        kappa.append(k)

        # k = cohen_kappa_score(np.sum(binarized_confusion_matrix, axis=1), np.sum(binarized_confusion_matrix, axis=0))
        # kappa.append(cohen_kappa_score(np.sum(binarized_confusion_matrix, axis=1), np.sum(binarized_confusion_matrix, axis=0)))

        if kappa[-1]<0:#not np.isnan(kappa[-1]):
            pdb.set_trace()

    return kappa

def cal_2afc(ann1, ann2):

    for _construct in range(4):
        for pair in range(2):
            if pair == 0:
                gt = ann1[:, 1+_construct]
                pred = ann2[:, 1+_construct]
            else:
                gt = ann2[:, 1+_construct]
                pred = ann1[:, 1+_construct]

            ndxPos = np.where(gt==1)[0]
            ndxNeg = np.where(gt==0)[0]

            v = 0

            for i in range(0, len(ndxPos)):
                if len(np.where(pred[ndxNeg]<pred[ndxPos[i]])[0]) > 0:
                    v += len(np.where(pred[ndxNeg]<pred[ndxPos[i]])[0])
                elif len(np.where(pred[ndxNeg] == pred[ndxPos[i]])[0]) > 0:
                    v += 0.5*len(np.where(pred[ndxNeg] == pred[ndxPos[i]])[0])
                # for j in range(0, len(ndxNeg)):
                #     if pred[ndxPos[i]] > pred[ndxNeg[j]]:
                #         v += 1
                #     elif pred[ndxPos[i]] == pred[ndxNeg[j]]:
                #         v += 0.5

            # score[:, ann1, ann2] = roc_auc_score(gt, pred)
            score[pair, _construct] =  v/(len(ndxNeg)*len(ndxPos))

    return score

if __name__ == '__main__':

    finished_videos = []
    window_len = 2

    for folder in os.listdir(LIFE_path):
        for char in ['Child', 'Parent']:
            conf_matrix = np.zeros((4, 4, 4, 4), dtype=np.int64)
            kappa = np.zeros((4, 4, 4), dtype=np.float64)

            # print(folder, '\n')
            if folder == 'Dyad' and DYAD == True:

                # print(char, '\n', os.listdir(os.path.join(LIFE_path, folder, 'window_' + str(WINDOW) + 'sec', char)))
                windowed_csvs = os.listdir(os.path.join(LIFE_path, folder, 'window_' + str(WINDOW) + 'sec', char))
                sub_folder = os.path.join(folder, 'window_' + str(WINDOW) + 'sec', char)

                for file in windowed_csvs:

                    frame_rate = 29.97
                    video_id, _, _, _annotator = file.split('_')
                    annotator_id = annotators[_annotator.split('.')[0]] # current annotator ID

                    # print('-' * 20)
                    # print(video_id)
                    # print('-' * 20)

                    file_ref = [(x, annotators[x.split('_')[-1].split('.')[0]]) if x.startswith(video_id) else None for x in windowed_csvs]
                    for ref in file_ref:
                        if ref is not None and ref[0] != file and ref[0] not in finished_videos: #  and file.startswith('2143212'):
                            _, data_array1, data_array2 = calculate_conf_matrix(sub_folder, file, ref[0], frame_rate, annotator_id, ref[1], nsecs=window_len)
                            finished_videos.append(ref[0])

                            if 'all_annotation1' not in locals():
                                all_annotation1 = data_array1
                            else:
                                all_annotation1 = np.append(all_annotation1, data_array1, axis=0)

                            if 'all_annotation2' not in locals():
                                all_annotation2 = data_array2
                            else:
                                all_annotation2 = np.append(all_annotation2, data_array2, axis=0)

                sio.savemat('EMO_afc_data.mat', {'ann1':all_annotation1, 'ann2':all_annotation2})

                # afc_score = cal_2afc(all_annotation1, all_annotation2)
                # print('afc_score-', afc_score)

                for a in range(conf_matrix.shape[0]):
                    for b in range(conf_matrix.shape[1]):
                        kappa[a, b, :] = calculate_kappa(conf_matrix[a, b, ...])
                kappa = np.reshape(kappa, (-1, 4))
                    # break
                # print('{0} {1}sec conf_matrix-{2} \n '.format(char, WINDOW, conf_matrix))
                print('{0} {1}sec \n '.format(char, WINDOW, conf_matrix))
                # print(kappa)
                print('kappa min-{0} \n mean-{1} \n max-{2} \n std-{3} \n'
                      .format(np.nanmin(kappa, axis=0), np.nanmean(kappa, axis=0), np.nanmax(kappa, axis=0), np.nanstd(kappa, axis=0)))
                # input('enter')



                # print(score)
                # sio.savemat('emo_'+char+'_kappa'+str(window_len), kappa)
                np.savez('emo_'+char+'_2afc'+str(float(window_len)), score)

            if folder == "Triad" and TRIAD == True:
                print(char, '\n', os.listdir(os.path.join(LIFE_path, folder, 'window_' + str(WINDOW) + 'sec', char)))

