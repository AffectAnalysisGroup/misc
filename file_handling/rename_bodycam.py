import os, pdb
import csv
import pandas as pd
# import cv2
from shutil import copyfile


body_cam_folder = '/run/user/1435715183/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Body_Camera/converted/EPI'
names_list = []

for video in os.listdir(body_cam_folder):
    if video.endswith('.mp4'):
        _temp = video.split('_')
        # print(_temp[0]+'_'+_temp[1]+'_'+_temp[-3]+'_'+_temp[-2]+'_'+_temp[-1])
        famID = _temp[0][-4:]
        # if _temp[1] == 'PSI':
        #     taskID = '02'
        # elif _temp[1] == 'EPI':
        #     taskID = '01'
        # elif _temp[2] == 'Neutral':
        #     taskID = '03'
        # else:
        #     taskID = '00'

        if 'PSI' in video:
            taskID = '02'
        elif 'EPI' in video:
            taskID = '01'
        elif 'neutral' in video.lower():
            taskID = '03'
        else:
            taskID = '00'



        subID = 1 if _temp[-1].split('.')[0] == '2' else 2
        # pdb.set_trace()
        new_name = famID + str(subID) + '_' + str(taskID) + '_01.mp4'
        # os.rename(video, new_name)
        # print(video, new_name)
        os.rename(os.path.join(body_cam_folder, video), os.path.join(body_cam_folder, new_name))
        if new_name in names_list:
            print('repetition-', new_name, video)
        else:
            names_list.append(new_name)
