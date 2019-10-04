import sys
# sys.path
# sys.path.insert(0, '/etc/VOLUME1/ADS_research_new/Custom_functions/')
import numpy as np
# import scipy.io as sio
from io import open
import os
import sys
import csv
import pdb
import Reading_data
import collections
import helper_function_LC


# for sub_sub_folder in SUB_SUB_FOLDER:
#
#     sys.stdout.flush()
#     ctr_file_count = 0
#
#     list_file_gt = sorted(os.listdir(ADS_construct_code_unpooled_folder + GT_SUB_FOLDER + sub_sub_folder))
#     list_file_rel = sorted(os.listdir(ADS_construct_code_unpooled_folder + REL_SUB_FOLDER + sub_sub_folder))
#
#     list_file_gt = [file for file in list_file_gt if len(file.split('.')) == 2]
#     list_file_rel = [file for file in list_file_rel if len(file.split('.')) == 2]
#
#     if len(list_file_gt) != len(list_file_rel):
#         pdb.set_trace()
#
#     for file_indx in range(len(list_file_gt)):
#         file_gt_path = ADS_construct_code_unpooled_folder + GT_SUB_FOLDER + sub_sub_folder + list_file_gt[file_indx]
#         file_rel_path = ADS_construct_code_unpooled_folder + REL_SUB_FOLDER + sub_sub_folder + list_file_rel[file_indx]
#
#         gt_data, _ = read_csv_file(file_gt_path)
#         rel_dat, _ = read_csv_file(file_rel_path)
#         rel_dat_pooled = np.zeros((rel_dat.shape))

def do_bhavan_code(rel_dat, gt_data, gt_orig_name, rel_orig_name, WINDOW_SIZE, ADS_FPS):

    nframe_pool = int(WINDOW_SIZE * ADS_FPS)

    rel_dat_pooled = np.zeros((rel_dat.shape))

    for ii in range(len(rel_dat)):

        if ii < len(gt_data):
            rel_dat_window = rel_dat[max(ii - nframe_pool, 0):ii + nframe_pool + 1, :]
            gt_value = gt_data[ii]

            for jj in range(len(gt_value)):

                if gt_value[jj] == 1:
                    rel_dat_pooled[ii, jj] = np.max(rel_dat_window[:, jj])
                elif gt_value[jj] == 0:
                    rel_dat_pooled[ii, jj] = np.min(rel_dat_window[:, jj])
                else:
                    pdb.set_trace()

    # gt_orig_name = file_gt_path.split('/')[-1]
    # rel_orig_name = file_rel_path.split('/')[-1]

    assert gt_orig_name.split('_')[0][1:-2] == rel_orig_name.split('_')[0][1:-2]

    gt_save_name = gt_orig_name.split('_')[0][0:-2] + '_' + gt_orig_name.split('_')[1] + '_' + \
                   gt_orig_name.split('_')[2][0:2] + '_' + gt_orig_name.split('_')[0][-2:] + '.csv'
    rel_save_name = rel_orig_name.split('_')[0][0:-2] + '_' + rel_orig_name.split('_')[1] + '_' + \
                    rel_orig_name.split('_')[2][0:2] + '_' + rel_orig_name.split('_')[0][-2:] + '.csv'

    print(gt_save_name)
    print(rel_save_name)

    gt_frame_no = np.arange(1, len(gt_data) + 1)  # gt_data
    rel_dat_pooled_frame_no = np.arange(1, len(rel_dat_pooled) + 1)  # rel_dat_pooled

    # if GT_SUB_FOLDER == 'cal/':
    #     gt_save_folder = ADS_construct_code_pooled_folder + sub_sub_folder
    #     rel_save_folder = ADS_construct_code_pooled_folder + sub_sub_folder
    # gt_save_folder  = ADS_construct_code_pooled_folder + 'cal/' + sub_sub_folder
    # rel_save_folder = ADS_construct_code_pooled_folder + 'rel/' + sub_sub_folder

    # else:
    # 	rel_save_folder = ADS_construct_code_pooled_folder + 'cal/' + sub_sub_folder
    # 	gt_save_folder  = ADS_construct_code_pooled_folder + 'rel/' + sub_sub_folder

    # if os.path.exists(gt_save_folder) == False:
    #     os.makedirs(gt_save_folder)
    #
    # if os.path.exists(rel_save_folder) == False:
    #     os.makedirs(rel_save_folder)

    # pdb.set_trace()

    # with open(gt_save_folder + gt_save_name, 'wb') as csvfile_write:
    #     csvwriter = csv.writer(csvfile_write)
    #     # csvwriter.writerow([0,82,83,84,85])
    #     csvwriter.writerow([0] + helper_function_LC.CODE_TO_COLUMN)
    #     for ii in range(len(gt_data)):
    #         csvwriter.writerow([gt_frame_no[ii]] + gt_data[ii].tolist())
    #
    # with open(rel_save_folder + rel_save_name, 'wb') as csvfile_write:
    #     csvwriter = csv.writer(csvfile_write)
    #     # csvwriter.writerow([0,82,83,84,85])
    #     csvwriter.writerow([0] + helper_function_LC.CODE_TO_COLUMN)
    #     for ii in range(len(rel_dat_pooled)):
    #         csvwriter.writerow([rel_dat_pooled_frame_no[ii]] + rel_dat_pooled[ii].tolist())

    # print(WINDOW_SIZE, ONSET_ONLY, sub_sub_folder)

    return