import os
import csv
import pandas as pd

body_cam_folder = '/run/user/1435715183/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Body Camera'

fp = open('bodycam_file.csv', 'w')
csvwriter = csv.writer(fp)

mapping_annotation_video = \
    pd.read_excel('DNT-TPOT_DataNamingMappingTable-VideoDataCamA.xlsx', skiprows=1, index_col=None).set_index('Original Filename')[
        'New Filename'].to_dict()

# print(mapping_annotation_video)

for video in os.listdir(body_cam_folder):
	if video.endswith('.ts'):
		_temp = video.split('_')
		# print(_temp[0]+'_'+_temp[1]+'_'+_temp[-3]+'_'+_temp[-2]+'_'+_temp[-1])
		print(video, mapping_annotation_video.get(_temp[0]+'_'+_temp[1]+'_'+_temp[-3]+'_'+_temp[-2]+'_'+_temp[-1], None))
		csvwriter.writerow([video, mapping_annotation_video.get(_temp[0]+'_'+_temp[1]+'_'+_temp[-3]+'_'+_temp[-2]+'_'+_temp[-1], 'not_found')])

		# print(video, mapping_annotation_video.get(video, None))
		# if 'PSI' in video:
		# 	video_new_name = video[:6]+'_'+'02_01'
		# 	csvwriter.writerow([video, video_new_name])

		# elif 'EPI' in video:
		# 	video_new_name = video[:6]+'_'+'01_01'

		# else:
		# 	csvwriter.writerow([video, 'Abnormal naming'])



		# _temp = video.split('_')
		# print(video, len(_temp))
		# if len(_temp) >= 7:
		# 	if len(_temp) == 6:
		# 		vid, task,_, _, _, _ = video.split('_')
		# 	elif len(_temp) == 7:
		# 		vid, task,_, _, _, _, _ = video.split('_')
		# 	else:
		# 		csvwriter.writerow([video, 'Abnormal naming'])
			
		# 	if task == 'EPI':
		# 		video_new_name = vid+'_'+'01'
		# 	elif task == 'PSI':
		# 		video_new_name = vid+'_'+'01'
		# 	csvwriter.writerow([video, video_new_name])
		# else:
		# 	csvwriter.writerow([video, 'Abnormal naming'])

fp.close()