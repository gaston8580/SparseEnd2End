import os
import mmcv
import tempfile
from utils import *
import cv2
import numpy as np

clips_path = '/home/ma-user/work/data/ali_odd'
for clip in os.listdir(clips_path):

    if clip!= 'clip_1730513152300':
        continue
    os.makedirs(os.path.join(clips_path, clip, 'vis_3d_lane_resize'), exist_ok=True)
    ##########Get the lane#########
    maps = mmcv.load(os.path.join(clips_path, 'label3d_line_lidarCS', 'LNNACDDV5PDA30339', clip, 'local_map',
                                  'line_3d_lidarCS.json'))
    #######Get Extrinsics and Intrinsics###
    info_json = mmcv.load(os.path.join(clips_path, clip, 'info.json'))
    plate_no = info_json['plate_no']
    with tempfile.TemporaryDirectory() as temp:
        exit_code = os.system(
            f'python /home/ma-user/work/calibration/load_calibration.py -v {plate_no} -p {clip} -c /home/ma-user/work/calibration/ -s {str(temp)}')
        if exit_code != 0:
            raise RuntimeError(f'{clip} get calibration failed !!!')
        ins_ex_path = str(temp)
        lidar2imu_path = os.path.join(ins_ex_path, 'extrinsics/lidar2imu/lidar2imu.yaml')
        with open(lidar2imu_path) as f:
            data_imu = yaml.safe_load(f)
        ins_ex = load_ins_ex(ins_ex_path)
    #####################
    sample_list = []
    for file in os.listdir(os.path.join(clips_path, clip)):
        if file.startswith('sample_'):
            sample_list.append(file)

    ##################
    for sample in sample_list:
        os.makedirs(os.path.join(clips_path, clip, 'vis_3d_lane_resize', sample), exist_ok=True)
        map_anno_lines = maps[sample]['lines']
        key_point_lines = []
        for line in map_anno_lines:
            line_tmp = []
            map_lines = line['line_key_points']
            for key_point in map_lines:

                line_tmp.append([key_point[0], key_point[1], key_point[2]])
                #if abs(key_point[0]) > 40 or abs(key_point[1]) > 40:
                #    continue
                #x_coords_line.append(key_point[0])
                #y_coords_line.append(key_point[1])
            key_point_lines.append(line_tmp)

        for sensor_file in os.listdir(os.path.join(clips_path, clip, sample)):
            if sensor_file.startswith('camera0') and sensor_file.find('undist') != -1:
                cam_type = 'camera0'
                img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))

                img_resize =  cv2.resize(img, dsize=(640, 480))
                    #img[int(1080-640):int(1080 + 640), int(1920 - 960):int(1920 + 960)]
                for i in range(len(key_point_lines)):
                    line = key_point_lines[i]
                    line_tmp = []
                    for j in range(len(line)):
                        x = line[j][0]
                        y = line[j][1]
                        z = line[j][2]
                        #if poly_shapely_gt.contains(point):
                        point_3d_homo = np.vstack((x, y, z, 1))
                        #####Cam0 Branch####
                        ###The extrinsics are the same###
                        lidar2cam_coords_cam0 = ins_ex[cam_type]['R_T'] @ point_3d_homo
                        # print(lidar2cam_coords[:3,:].shape)
                        intrinsics = ins_ex[cam_type]['K']
                        #print(intrinsics)
                        scale = np.eye(3)

                        scale[0][0] = float(640 / img.shape[1])
                        scale[1][1] = float(480 / img.shape[0])
                        intrinsics_new = scale @ intrinsics
                        uv_coords_cam0 =  intrinsics_new @ lidar2cam_coords_cam0[:3, :]
                        u_cam0, v_cam0, z_cam0 = uv_coords_cam0[0][0], uv_coords_cam0[1][0], uv_coords_cam0[2][0]
                        if z_cam0 > 0:
                            # print(z_cam)
                            u_norm_cam0 = float(u_cam0 / z_cam0)
                            v_norm_cam0 = float(v_cam0 / z_cam0)
                            if u_norm_cam0 >= 0 and u_norm_cam0 <= img_resize.shape[1] and v_norm_cam0 >= 0 and v_norm_cam0 <= img_resize.shape[0]:
                                print(u_norm_cam0, v_norm_cam0)
                                line_tmp.append([u_norm_cam0, v_norm_cam0])
                                #cv2.circle(img_resize, (int(u_norm_cam0), int(v_norm_cam0)), 3, (255, 0, 0), 3)
                    color = (255, 0, 0)  # Blue color in BGR
                    thickness = 1
                    isClosed = False
                    print(np.array(line_tmp).reshape(-1,1,2))
                    img_resize = cv2.polylines(img_resize, [np.array(line_tmp).reshape(-1,1,2)], isClosed, color, thickness)
                cv2.imwrite(os.path.join(clips_path, clip, 'vis_3d_lane_resize', sample, sensor_file), img_resize)

