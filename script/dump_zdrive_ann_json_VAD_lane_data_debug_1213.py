import mmcv
import json
import os
import numpy as np
from pypcd import pypcd
from pyquaternion import Quaternion
import tempfile
import yaml
from utils import *
from scipy.spatial.transform import Rotation as Rot
from pyquaternion import Quaternion
import math
from scipy import interpolate
import cv2

def quaternion_to_euler(q):
    (w, x, y, z) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw, pitch, roll
def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw



def main():


    #clips_path = '/media/com0109/d326c99f-2622-4ca8-b10c-1cb533e255af/dataset/Pilot3.3_VAD/Pilot3.3_VAD'
    clips_path = '/home/ma-user/work/data/ali_odd'
    cat2idx = {}
    category = ['car', 'truck', 'construction_vehicle', 'bus', 'tricycle', 'bicycle', 'motorcycle', 'person']
    for idx, dic in enumerate(category):
        cat2idx[dic] = idx

    #infos = []
    for clip in os.listdir(clips_path):
        infos = []
        print(clip)
        if not clip.startswith('clip'):
            continue
        if clip != 'clip_1730513132300':
            continue
        if clip.endswith('.json'):
            continue
        sample_list = []
        for file in os.listdir(os.path.join(clips_path, clip)):
            if file.startswith('sample_'):
                sample_list.append(file)

        sample_list.sort()
        ####Read Poses###

        pose_json = mmcv.load(os.path.join(clips_path, clip, 'localization.json'))
        #root_path + date + '/' + clip + '/localization.json')

        ####Read Mod annos###
        with open(os.path.join(clips_path, clip + '.json')) as f:
            g_total  = json.load(f)



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

        lidar2ego_trans = [data_imu['transform']['translation']['x'], data_imu['transform']['translation']['y'], data_imu['transform']['translation']['z']]
        lidar2ego_rots = [data_imu['transform']['rotation']['w'],data_imu['transform']['rotation']['x'], data_imu['transform']['rotation']['y'],data_imu['transform']['rotation']['z']]
        #######################################
        #maps = mmcv.load(os.path.join(clips_path, clip, 'localmap', 'line_3d_lidarCS.json'))
        maps = mmcv.load(os.path.join(clips_path, 'label3d_line_lidarCS','LNNACDDV5PDA30339', clip, 'local_map', 'line_3d_lidarCS.json'))
        for sample in sample_list:
            #####Lod MOD/GOP Annos#######
            anno_file = sample[7:17] + '.' +sample[17:] +'.json'
            #if not os.path.exists(os.path.join(clips_path, 'output', clip, 'lidar_obstacle_3d_refined', anno_file)):
            #    continue
            #with open(os.path.join(clips_path, 'output', clip, 'lidar_obstacle_3d_refined', anno_file)) as f:

            info = {}
            info['token'] = clip + sample
            info['scene_token'] = clip


            ################Get the lane coordinates#################
            map_anno_lines = maps[sample]['lines']
            lines= []
            for line in map_anno_lines:
                line_tmp = []
                map_lines = line['line_key_points']
                x_coords_line = []
                y_coords_line = []
                for key_point in map_lines:
                    if abs(key_point[0]) > 40 or abs(key_point[1]) > 40:
                      continue
                    x_coords_line.append(key_point[0])
                    y_coords_line.append(key_point[1])
                    line_tmp.append([key_point[0], key_point[1]])
                if len(line_tmp) <= 1:
                    continue
                '''
                x_coords_line = np.array(x_coords_line)
                y_coords_line = np.array(y_coords_line)
                f_interp = interpolate.interp1d(x_coords_line, y_coords_line)
                
                x_new_coords = np.linspace(x_coords_line.min(), x_coords_line.max(), num=200, endpoint=True)
                y_new_coords = f_interp(x_new_coords)
                for x_new_coord, y_new_coord in zip(x_new_coords, y_new_coords):
                    if x_new_coord < 120 and x_new_coord > -60 and abs(y_new_coord)< 60:
                        line_tmp.append([x_new_coord, y_new_coord])
                '''
                lines.append(line_tmp)
            dict_map = {}
            dict_map["0"] = lines
            info['map_annos'] = dict_map
            ###########Obtain the pose information############
            digit_length = len(sample.split('sample_')[-1])
            # print(len(sample.split('sample_')[-1]))
            time_stamp_sample = int(sample.split('sample_')[-1])
            # time_stamp_sample = int(g['frames'][i]['frame_name'].split('sample_')[-1])
            upper_bound = 10000000000000000
            for j in range(len(pose_json)):
                # print(pose_json[j]['frame_name'])
                # time_stamp_pose = int(pose_json[j]['frame_name'].split('sample_')[-1])
                # print("Pose TimeStamp")
                # print(pose_json[j]['timestamp'])
                time_stamp_pose = 1000000 * float(pose_json[j]['timestamp'])
                # time_stamp_pose = pow(10, (16-digit_length)) * float(pose_json[j]['timestamp'])
                # if pose_json[j]['frame_name']==sample:
                if abs(time_stamp_sample - time_stamp_pose) < upper_bound:
                    # print(pose_json[j]['frame_name'])
                    pose_x = pose_json[j]['pose']['position']['x']
                    pose_y = pose_json[j]['pose']['position']['y']
                    pose_z = pose_json[j]['pose']['position']['z']

                    pose_qw = pose_json[j]['pose']['orientation']['qw']
                    pose_qx = pose_json[j]['pose']['orientation']['qx']
                    pose_qy = pose_json[j]['pose']['orientation']['qy']
                    pose_qz = pose_json[j]['pose']['orientation']['qz']
                    upper_bound = abs(time_stamp_sample - time_stamp_pose)
            ###############################################
            info['lidar2ego_translation'] = lidar2ego_trans  # [0.027712,1.289104, -0.337598]
            info['lidar2ego_rotation'] = lidar2ego_rots
            info['ego2global_translation'] = [pose_x, pose_y, pose_z]
            info['ego2global_rotation'] = [pose_qw, pose_qx, pose_qy, pose_qz]



            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            e2g_r = info['ego2global_rotation']
            e2g_t = info['ego2global_translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix
            info['timestamp'] = time_stamp_sample



            # sample_path = root_path+ date + '/' + clip + '/' + sample

            #######Get the Lidar Path #########
            sample_path = os.path.join(clips_path, clip, sample)
            sensor_files = os.listdir(sample_path)
            for sensor_file in sensor_files:
                # if sensor_file.find('lidar0') != -1 and sensor_file.find('mc') != -1 and sensor_file.find('.pcd')!=-1:
                # if sensor_file.find('lidar0') != -1 and sensor_file.find('.bin') == -1:
                if sensor_file.find('lidar0') != -1 and sensor_file.find('.pcd') != -1:
                    sensor_fname = sensor_file.split('.')[0] + '.bin'
                    sensor_fname_pcd = sensor_file
                    # pc = pypcd.PointCloud.from_path(folder_path_new + '/' + sensor_file)
                    # print(sample_path + '/' + sensor_file)
                    pc = pypcd.PointCloud.from_path(sample_path + '/' + sensor_file)
            # print((pc.pc_data.dtype))
            ## Get data from pcd (x, y, z, intensity, ring, time)

            np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
            np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
            # np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
            np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
            # np_i = (np.array(pc.pc_data['rgb'], dtype=np.float32)).astype(np.float32)/256
            # print(np.array(pc.pc_data['intensity'], dtype=np.float32))
            np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32) / 255.0
            np_t = (np.array(pc.pc_data['timestamp'], dtype=np.float32)).astype(np.float32)
            np_r = (np.array(pc.pc_data['ring'], dtype=np.float32)).astype(np.float32)
            # bin_file_path = folder_path_new + '/' + sensor_fname
            bin_file_path = os.path.join(clips_path, clip, sample, sensor_fname )
            ## Stack all data
            points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i, np_r)))

            ## Save bin file
            points_32.tofile(bin_file_path)
            # print("File is saved to")
            # print(bin_file_path)
            bin_file_path = os.path.join(clips_path, clip, sample, sensor_fname)
            info['lidar_path'] = bin_file_path
            index_i = sample_list.index(sample)

            ##############Get MultiSweeps #########################
            num_sweeps = 5
            sweeps = []
            #index_i = sample_list.index(sample)


            if index_i == 0:
                info['sweeps'] = sweeps
            else:
                for kk in range(-1, -num_sweeps, -1):
                    if index_i + kk >= 0:
                        sample_sweep = sample_list[index_i + kk]
                        digits_len_sweep = len(sample_sweep.split('sample_')[-1])
                        time_stamp_sample_sweep = int(sample_sweep.split('sample_')[-1])
                        upper_bound_sweep = 10000000000000000
                        # print("Pose Timestamp")
                        for k in range(len(pose_json)):
                            time_stamp_pose_sweep = 1000000 * float(pose_json[k]['timestamp'])
                            # time_stamp_pose_sweep = pose_json[k]['timestamp']
                            # if pose_json[j]['frame_name']==sample:
                            # print()
                            if abs(time_stamp_sample_sweep - time_stamp_pose_sweep) < upper_bound_sweep:
                                # print(time_stamp_sample_sweep)
                                # print(time_stamp_pose_sweep)
                                pose_x_sweep = pose_json[k]['pose']['position']['x']
                                pose_y_sweep = pose_json[k]['pose']['position']['y']
                                pose_z_sweep = pose_json[k]['pose']['position']['z']

                                pose_qw_sweep = pose_json[k]['pose']['orientation']['qw']
                                pose_qx_sweep = pose_json[k]['pose']['orientation']['qx']
                                pose_qy_sweep = pose_json[k]['pose']['orientation']['qy']
                                pose_qz_sweep = pose_json[k]['pose']['orientation']['qz']
                                upper_bound_sweep = abs(time_stamp_sample_sweep - time_stamp_pose_sweep)
                                # print("Upper Bound")
                                # print(upper_bound_sweep)

                        # sample_sweep_path = root_path +'/' + date +'/'+ clip + '/' + sample_sweep
                        sample_sweep_path = os.path.join(clips_path, clip, sample_sweep)
                        #root_path + '/' + date + clip + '/' + sample_sweep

                        sensor_sweep_files = os.listdir(sample_sweep_path)
                        for sensor_sweep_file in sensor_sweep_files:
                            # if sensor_sweep_file.find('lidar0') != -1 and sensor_sweep_file.find('mc') != -1 and sensor_sweep_file.find('.pcd')!=-1:
                            # if sensor_sweep_file.find('lidar0') != -1 and sensor_sweep_file.find('.bin')==-1:
                            if sensor_sweep_file.find('lidar0') != -1 and sensor_sweep_file.find('.pcd') != -1:
                                sensor_sweep_fname = sensor_sweep_file.split('.')[0]

                        sweep = {
                            'data_path': os.path.join(clips_path, clip, sample_sweep, sensor_sweep_fname + '.bin'),
                            #'./data/Pegasus_Multisweeps/' + date + '/' + clip + '/' + sample_sweep + '/' + sensor_sweep_fname + '.bin',
                            # 'sensor2ego_translation': [0.027712,1.289104,1.642402],
                            'sensor2ego_translation': lidar2ego_trans,
                            # 'sensor2ego_translation': [1.204, 0.0, -0.776],
                            # cs_record['translation'],
                            # 'sensor2ego_rotation': [0.008842517215726329, 0.9999418715902747, -0.0044230909277731675, 0.004301115724160793],
                            'sensor2ego_rotation': lidar2ego_rots,
                            # [0.005499,0.009583,0.716402,0.697601],
                            # cs_record['rotation'],
                            'ego2global_translation': [pose_x_sweep, pose_y_sweep, pose_z_sweep],
                            # pose_record['translation'],
                            'timestamp': time_stamp_sample_sweep,
                            'ego2global_rotation': [pose_qw_sweep, pose_qx_sweep, pose_qy_sweep, pose_qz_sweep]}
                        #####Save bin samples
                        pc_sweep = pypcd.PointCloud.from_path(
                            os.path.join(clips_path, clip, sample_sweep, sensor_sweep_fname+'.pcd')
                            #root_path + date + '/' + clip + '/' + sample_sweep + '/' + sensor_sweep_fname + '.pcd'
                            )
                        np_x_sweep = (np.array(pc_sweep.pc_data['x'], dtype=np.float32)).astype(np.float32)
                        np_y_sweep = (np.array(pc_sweep.pc_data['y'], dtype=np.float32)).astype(np.float32)
                        np_z_sweep = (np.array(pc_sweep.pc_data['z'], dtype=np.float32)).astype(np.float32)
                        np_i_sweep = (np.array(pc_sweep.pc_data['intensity'], dtype=np.float32)).astype(
                            np.float32) / 255.0
                        np_t_sweep = (np.array(pc_sweep.pc_data['timestamp'], dtype=np.float32)).astype(
                            np.float32) / 255.0
                        points_32_sweep = np.transpose(
                            np.vstack((np_x_sweep, np_y_sweep, np_z_sweep, np_i_sweep, np_t_sweep)))

                        points_32_sweep.tofile(
                            #root_path + date + '/' + clip + '/' + sample_sweep + '/' + sensor_sweep_fname + '.bin'
                            os.path.join(clips_path, clip, sample_sweep, sensor_sweep_fname +'.bin')
                            )
                        ###Sensor2Lidar Pose#########
                        l2e_r_s = sweep['sensor2ego_rotation']
                        l2e_t_s = sweep['sensor2ego_translation']
                        e2g_r_s = sweep['ego2global_rotation']
                        e2g_t_s = sweep['ego2global_translation']
                        # obtain the RT from sensor to Top LiDAR
                        # sweep->ego->global->ego'->lidar
                        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
                        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                                      ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                        sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
                        sweep['sensor2lidar_translation'] = T
                        sweeps.append(sweep)
            info['sweeps'] = sweeps

            #############Get the  Cam Branch ######################
            camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_WIDE'
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
            info['cams'] = {}
            info['cams']['CAM_FRONT_WIDE'] = {}
            info['cams']['CAM_FRONT'] = {}
            info['cams']['CAM_FRONT_LEFT'] = {}
            info['cams']['CAM_FRONT_RIGHT'] = {}
            info['cams']['CAM_BACK_LEFT'] = {}
            info['cams']['CAM_BACK_RIGHT'] = {}
            info['cams']['CAM_BACK'] = {}
            ######################
            for sensor_file in os.listdir(os.path.join(clips_path, clip, sample)):
                if sensor_file.startswith('camera0') and sensor_file.find('undist')==-1:
                    print(sensor_file)
                    info['cams']['CAM_FRONT_WIDE']['data_path'] = os.path.join(clips_path, clip, sample, sensor_file)
                    lidar2cam_extrinsics = ins_ex['camera0']['R_T']
                    cam2lidar_extrinsics = np.linalg.inv(ins_ex['camera0']['R_T'])

                    intrinsics = ins_ex['camera0']['K']
                    intrinsics[0][0] = 1473.0
                    intrinsics[1][1] = 1473.0


                    info['cams']['CAM_FRONT_WIDE']['ego2global_translation'] = [pose_x, pose_y, pose_z]
                    info['cams']['CAM_FRONT_WIDE']['ego2global_rotation'] =  [pose_qw, pose_qx, pose_qy, pose_qz]

                    #########
                    info['cams']['CAM_FRONT_WIDE']['sensor2lidar_rotation'] =  cam2lidar_extrinsics[:3, :3]
                    info['cams']['CAM_FRONT_WIDE']['sensor2lidar_translation'] = cam2lidar_extrinsics[:3, 3]

                    sensor2ego = ins_ex['lidar2imu'] @ cam2lidar_extrinsics
                    info['cams']['CAM_FRONT_WIDE']['sensor2ego_rotation'] =  [
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[3],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[0],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[1],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[2]]
                    info['cams']['CAM_FRONT_WIDE']['sensor2ego_translation'] = sensor2ego[:3, 3]
                    raw_img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                    scale = np.eye(3)
                    scale[0][0]= float(640/ raw_img.shape[1])
                    scale[1][1]= float(480/ raw_img.shape[0])
                    info['cams']['CAM_FRONT_WIDE']['cam_intrinsic'] = scale@ intrinsics
                    info['cams']['CAM_FRONT_WIDE']['cam_distortion'] = ins_ex['camera0']['K']
                    info['cams']['CAM_FRONT_WIDE']['timestamp'] = time_stamp_sample

                if sensor_file.startswith('camera1') and sensor_file.find('undist')==-1 and sensor_file.startswith('camera10')==False:
                    info['cams']['CAM_FRONT']['data_path'] = os.path.join(clips_path, clip, sample, sensor_file)

                    cam2lidar_extrinsics = np.linalg.inv(ins_ex['camera1']['R_T'])
                    intrinsics = ins_ex['camera1']['K']

                    info['cams']['CAM_FRONT']['ego2global_translation'] = [pose_x, pose_y, pose_z]
                    info['cams']['CAM_FRONT']['ego2global_rotation'] = [pose_qw, pose_qx, pose_qy, pose_qz]

                    info['cams']['CAM_FRONT']['sensor2lidar_rotation'] = cam2lidar_extrinsics[:3, :3]
                    info['cams']['CAM_FRONT']['sensor2lidar_translation'] = cam2lidar_extrinsics[:3, 3]
                    sensor2ego = ins_ex['lidar2imu'] @ cam2lidar_extrinsics
                    info['cams']['CAM_FRONT']['sensor2ego_rotation'] = [
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[3],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[0],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[1],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[2]]
                    info['cams']['CAM_FRONT']['sensor2ego_translation'] = sensor2ego[:3, 3]
                    raw_img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                    scale = np.eye(3)
                    scale[0][0] = float(640 / raw_img.shape[1])
                    scale[1][1] = float(480 / raw_img.shape[0])
                    info['cams']['CAM_FRONT']['cam_intrinsic'] = scale@intrinsics
                    info['cams']['CAM_FRONT']['cam_distortion'] = ins_ex['camera1']['K']
                    info['cams']['CAM_FRONT']['timestamp'] = time_stamp_sample
                if sensor_file.startswith('camera2') and sensor_file.find('undist')==-1:
                    info['cams']['CAM_FRONT_LEFT']['data_path'] = os.path.join(clips_path, clip, sample, sensor_file)
                    cam2lidar_extrinsics = np.linalg.inv(ins_ex['camera2']['R_T'])
                    intrinsics = ins_ex['camera2']['K']

                    info['cams']['CAM_FRONT_LEFT']['ego2global_translation'] = [pose_x, pose_y, pose_z]
                    info['cams']['CAM_FRONT_LEFT']['ego2global_rotation'] = [pose_qw, pose_qx, pose_qy, pose_qz]

                    info['cams']['CAM_FRONT_LEFT']['sensor2lidar_rotation'] = cam2lidar_extrinsics[:3, :3]
                    info['cams']['CAM_FRONT_LEFT']['sensor2lidar_translation'] = cam2lidar_extrinsics[:3, 3]

                    sensor2ego = ins_ex['lidar2imu'] @ cam2lidar_extrinsics
                    info['cams']['CAM_FRONT_LEFT']['sensor2ego_rotation'] = [
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[3],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[0],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[1],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[2]]
                    info['cams']['CAM_FRONT_LEFT']['sensor2ego_translation'] = sensor2ego[:3, 3]
                    raw_img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                    scale = np.eye(3)
                    scale[0][0] = float(640 / raw_img.shape[1])
                    scale[1][1] = float(480 / raw_img.shape[0])
                    info['cams']['CAM_FRONT_LEFT']['cam_intrinsic'] = scale@intrinsics
                    info['cams']['CAM_FRONT_LEFT']['cam_distortion'] = ins_ex['camera2']['K']
                    info['cams']['CAM_FRONT_LEFT']['timestamp'] = time_stamp_sample

                if sensor_file.startswith('camera3') and sensor_file.find('undist')==-1:
                    info['cams']['CAM_BACK_LEFT']['data_path'] = os.path.join(clips_path, clip, sample, sensor_file)
                    cam2lidar_extrinsics = np.linalg.inv(ins_ex['camera3']['R_T'])
                    intrinsics = ins_ex['camera3']['K']

                    info['cams']['CAM_BACK_LEFT']['ego2global_translation'] = [pose_x, pose_y, pose_z]
                    info['cams']['CAM_BACK_LEFT']['ego2global_rotation'] = [pose_qw, pose_qx, pose_qy, pose_qz]

                    info['cams']['CAM_BACK_LEFT']['sensor2lidar_rotation'] = cam2lidar_extrinsics[:3, :3]
                    info['cams']['CAM_BACK_LEFT']['sensor2lidar_translation'] = cam2lidar_extrinsics[:3, 3]

                    sensor2ego = ins_ex['lidar2imu'] @ cam2lidar_extrinsics
                    info['cams']['CAM_BACK_LEFT']['sensor2ego_rotation'] = [
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[3],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[0],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[1],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[2]]
                    info['cams']['CAM_BACK_LEFT']['sensor2ego_translation'] = sensor2ego[:3, 3]
                    raw_img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                    scale = np.eye(3)
                    scale[0][0] = float(640 / raw_img.shape[1])
                    scale[1][1] = float(480 / raw_img.shape[0])
                    info['cams']['CAM_BACK_LEFT']['cam_intrinsic'] = scale@intrinsics
                    info['cams']['CAM_BACK_LEFT']['cam_distortion'] = ins_ex['camera3']['K']
                    info['cams']['CAM_BACK_LEFT']['timestamp'] = time_stamp_sample

                if sensor_file.startswith('camera4') and sensor_file.find('undist')==-1:
                    info['cams']['CAM_FRONT_RIGHT']['data_path'] = os.path.join(clips_path, clip, sample, sensor_file)

                    cam2lidar_extrinsics = np.linalg.inv(ins_ex['camera4']['R_T'])
                    intrinsics = ins_ex['camera4']['K']

                    info['cams']['CAM_FRONT_RIGHT']['ego2global_translation'] = [pose_x, pose_y, pose_z]
                    info['cams']['CAM_FRONT_RIGHT']['ego2global_rotation'] = [pose_qw, pose_qx, pose_qy, pose_qz]

                    info['cams']['CAM_FRONT_RIGHT']['sensor2lidar_rotation'] = cam2lidar_extrinsics[:3, :3]
                    info['cams']['CAM_FRONT_RIGHT']['sensor2lidar_translation'] = cam2lidar_extrinsics[:3, 3]

                    sensor2ego = ins_ex['lidar2imu'] @ cam2lidar_extrinsics
                    info['cams']['CAM_FRONT_RIGHT']['sensor2ego_rotation'] = [
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[3],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[0],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[1],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[2]]
                    info['cams']['CAM_FRONT_RIGHT']['sensor2ego_translation'] = sensor2ego[:3, 3]
                    raw_img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                    scale = np.eye(3)
                    scale[0][0] = float(640 / raw_img.shape[1])
                    scale[1][1] = float(480 / raw_img.shape[0])
                    info['cams']['CAM_FRONT_RIGHT']['cam_intrinsic'] = scale@intrinsics
                    info['cams']['CAM_FRONT_RIGHT']['cam_distortion'] = ins_ex['camera4']['K']
                    info['cams']['CAM_FRONT_RIGHT']['timestamp'] = time_stamp_sample

                if sensor_file.startswith('camera5') and sensor_file.find('undist')==-1:
                    info['cams']['CAM_BACK_RIGHT']['data_path'] = os.path.join(clips_path, clip, sample, sensor_file)

                    cam2lidar_extrinsics = np.linalg.inv(ins_ex['camera5']['R_T'])
                    intrinsics = ins_ex['camera5']['K']

                    info['cams']['CAM_BACK_RIGHT']['ego2global_translation'] = [pose_x, pose_y, pose_z]
                    info['cams']['CAM_BACK_RIGHT']['ego2global_rotation'] = [pose_qw, pose_qx, pose_qy, pose_qz]

                    info['cams']['CAM_BACK_RIGHT']['sensor2lidar_rotation'] = cam2lidar_extrinsics[:3, :3]
                    info['cams']['CAM_BACK_RIGHT']['sensor2lidar_translation'] = cam2lidar_extrinsics[:3, 3]

                    sensor2ego = ins_ex['lidar2imu'] @ cam2lidar_extrinsics
                    info['cams']['CAM_BACK_RIGHT']['sensor2ego_rotation'] = [
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[3],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[0],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[1],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[2]]
                    info['cams']['CAM_BACK_RIGHT']['sensor2ego_translation'] = sensor2ego[:3, 3]
                    raw_img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                    scale = np.eye(3)
                    scale[0][0] = float(640 / raw_img.shape[1])
                    scale[1][1] = float(480 / raw_img.shape[0])
                    info['cams']['CAM_BACK_RIGHT']['cam_intrinsic'] = scale@intrinsics
                    info['cams']['CAM_BACK_RIGHT']['cam_distortion'] = ins_ex['camera5']['K']
                    info['cams']['CAM_BACK_RIGHT']['timestamp'] = time_stamp_sample

                if sensor_file.startswith('camera6') and sensor_file.find('undist')==-1:
                    info['cams']['CAM_BACK']['data_path'] = os.path.join(clips_path, clip, sample, sensor_file)
                    cam2lidar_extrinsics = np.linalg.inv(ins_ex['camera6']['R_T'])
                    intrinsics = ins_ex['camera6']['K']

                    info['cams']['CAM_BACK']['ego2global_translation'] = [pose_x, pose_y, pose_z]
                    info['cams']['CAM_BACK']['ego2global_rotation'] = [pose_qw, pose_qx, pose_qy, pose_qz]

                    info['cams']['CAM_BACK']['sensor2lidar_rotation'] = cam2lidar_extrinsics[:3, :3]
                    info['cams']['CAM_BACK']['sensor2lidar_translation'] = cam2lidar_extrinsics[:3, 3]

                    sensor2ego = ins_ex['lidar2imu'] @ cam2lidar_extrinsics
                    info['cams']['CAM_BACK']['sensor2ego_rotation'] = [
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[3],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[0],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[1],
                        Rot.from_matrix(sensor2ego[:3, :3]).as_quat()[2]]
                    info['cams']['CAM_BACK']['sensor2ego_translation'] = sensor2ego[:3, 3]
                    raw_img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                    scale = np.eye(3)
                    scale[0][0] = float(640 / raw_img.shape[1])
                    scale[1][1] = float(480 / raw_img.shape[0])
                    info['cams']['CAM_BACK']['cam_intrinsic'] = scale@intrinsics
                    info['cams']['CAM_BACK']['cam_distortion'] = ins_ex['camera6']['K']
                    info['cams']['CAM_BACK']['timestamp'] = time_stamp_sample


            ############Prepare for the  MOD and VAD###################
            boxes_list = []
            labels_list = []
            track_list = []
            velocity_list = []

            for g in g_total['frames']:
                if g['frame_name'] == sample:
                    g_tmp = g


            objs  =  g_tmp['annotated_info']['3d_city_object_detection_annotated_info']['annotated_info']['3d_object_detection_info']['3d_object_detection_anns_info']
            for obj in objs:
                if abs(obj['obj_center_pos'][0]) < 0.8 and abs(obj['obj_center_pos'][1]) < 0.8:
                    continue

                yaw_obj,_, _  = \
                    quaternion_to_euler([float(obj['obj_rotation'][3]), float(obj['obj_rotation'][0]), float(obj['obj_rotation'][1]), float(obj['obj_rotation'][2])])
                bbox = [obj['obj_center_pos'][0], obj['obj_center_pos'][1], obj['obj_center_pos'][2],
                        obj['size'][0], obj['size'][1], obj['size'][2],
                        yaw_obj,
                        obj['velocity'][0], obj['velocity'][1]
                        ]
                '''
                bbox = [obj['psr']['position']['x'], obj['psr']['position']['y'], obj['psr']['position']['z'],
                        obj['psr']['scale']['x'], obj['psr']['scale']['y'], obj['psr']['scale']['z'],
                        obj['psr']['rotation']['z'],
                        #obj['utm_velocity']['x'], obj['utm_velocity']['y']
                        ]
                '''
                #velocity_list.append([obj['utm_velocity']['x'], obj['utm_velocity']['y']])
                velocity_list.append([obj['velocity'][0], obj['velocity'][1]])


                boxes_list.append(bbox)
                labels_list.append(obj['category'])
                track_list.append(obj['track_id'])
            info["instance_inds"] =  track_list
            gt_boxes = np.array(boxes_list)
            gt_names = np.array(labels_list)
            gt_velocity = np.array(velocity_list)
            # get future coords for each box
            # [num_box, fut_ts*2]
            ##Futs_ts is 3 second#
            fut_ts = 6*5
            ####################
            fut_valid_flag = True
            for kkk in range(fut_ts):
                if index_i +  kkk>= len(sample_list)-1:

                    fut_valid_flag = False
            info['fut_valid_flag'] = fut_valid_flag
            num_box = len(boxes_list)
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
            gt_fut_yaw = np.zeros((num_box, fut_ts))
            gt_fut_masks = np.zeros((num_box, fut_ts))
            gt_boxes_yaw = gt_boxes[:, 6]
            #agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)

            # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
            agent_lcf_feat = np.zeros((num_box, 9))
            gt_fut_goal = np.zeros((num_box))
            #for i, anno in enumerate(annotations):
            for i , bbox in enumerate(gt_boxes):
                cur_box = bbox
                #cur_anno = anno
                agent_lcf_feat[i, 0:2] = cur_box[:2]
                agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
                agent_lcf_feat[i, 3:5] = gt_velocity[i]
                agent_lcf_feat[i, 5:8] = cur_box[3:6]
                agent_lcf_feat[i, 8] = cat2idx[gt_names[i]]
                for j in range(fut_ts):
                    index_next =index_i +j
                    if index_next<=len(sample_list)-1:
                        sample_next = sample_list[index_next]

                        for g in g_total['frames']:
                            if g['frame_name'] == sample_next:
                                g_tmp_next = g

                        objs_next = g_tmp_next['annotated_info']['3d_city_object_detection_annotated_info']['annotated_info'][
                            '3d_object_detection_info']['3d_object_detection_anns_info']
                        #objs_next =mmcv.load(os.path.join(clips_path, 'output', clip, 'lidar_obstacle_3d_refined',  sample_list[index_next][7:17] +'.'+ sample_list[index_next][17:] +'.json'))
                        #print(os.path.join(clips_path, 'output', clip, 'lidar_obstacle_3d_refined',  sample_list[index_next][7:17] +'.'+ sample_list[index_next][17:] +'.json'))
                        flag_has_next = False
                        '''
                        for obj_next in objs_next['objects']['lidar0']:
                            if obj_next['obj_id'] == track_list[i]:
                                flag_has_next =True
                                obj_next_tmp = obj_next
                        '''
                        for obj_next in objs_next:
                            if obj_next['track_id'] == track_list[i]:
                                #flag_has_next = True
                                obj_next_tmp = obj_next
                                if abs(obj_next_tmp['obj_center_pos'][0]) < 0.8 and abs(obj_next_tmp['obj_center_pos'][1]) < 0.8:
                                    flag_has_next = True


                        if flag_has_next == True:
                             yaw_obj_next_tmp, _, _ = \
                                quaternion_to_euler(
                                    [obj_next_tmp['obj_rotation'][3], obj_next_tmp['obj_rotation'][0], obj_next_tmp['obj_rotation'][1],
                                     obj_next_tmp['obj_rotation'][2]])

                             box_next = [obj_next_tmp['obj_center_pos'][0], obj_next_tmp['obj_center_pos'][1], obj_next_tmp['obj_center_pos'][2],
                                        obj_next_tmp['size'][0], obj_next_tmp['size'][1], obj_next_tmp['size'][2],
                                        yaw_obj_next_tmp,
                                        obj_next_tmp['velocity'][0], obj_next_tmp['velocity'][1]
                                        ]

                             #box_next = [obj_next_tmp['psr']['position']['x'], obj_next_tmp['psr']['position']['y'], obj_next_tmp['psr']['position']['z'],
                             #obj_next_tmp['psr']['scale']['x'], obj_next_tmp['psr']['scale']['y'], obj_next_tmp['psr']['scale']['z'],
                             #obj_next_tmp['psr']['rotation']['z'],
                             #obj_next_tmp['utm_velocity']['x'], obj_next_tmp['utm_velocity']['y']
                             #]
                             gt_fut_trajs[i, j] = np.array(box_next[:2]) - np.array(cur_box[:2])
                             gt_fut_masks[i, j] = 1
                             gt_fut_yaw[i, j] = box_next[6] - cur_box[6]
                             cur_box = box_next

                        else:
                            gt_fut_trajs[i, j:] = 0
                            break

                # get agent goal
                gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
                coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
                if coord_diff.max() < 1.0:  # static
                    gt_fut_goal[i] = 9
                else:
                    box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                    gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class

            # get ego history traj (offset format)
            his_ts = 10
            ego_his_trajs = np.zeros((his_ts + 1, 3))
            ego_his_trajs_diff = np.zeros((his_ts + 1, 3))
            #sample_cur = sample
            for index_his in range(his_ts, -1, -1):
                ####If Cur Sample Exists##
                if index_i + index_his -his_ts >=0:

                    #######Get Pose#######
                    time_stamp_sample_his = int(sample_list[index_i + index_his - his_ts].split('sample_')[-1])
                    # time_stamp_sample = int(g['frames'][i]['frame_name'].split('sample_')[-1])
                    upper_bound = 10000000000000000
                    for j in range(len(pose_json)):
                        # print(pose_json[j]['frame_name'])
                        # time_stamp_pose = int(pose_json[j]['frame_name'].split('sample_')[-1])
                        # print("Pose TimeStamp")
                        # print(pose_json[j]['timestamp'])
                        time_stamp_pose = 1000000 * float(pose_json[j]['timestamp'])
                        # time_stamp_pose = pow(10, (16-digit_length)) * float(pose_json[j]['timestamp'])
                        # if pose_json[j]['frame_name']==sample:
                        if abs(time_stamp_sample_his - time_stamp_pose) < upper_bound:
                            # print(pose_json[j]['frame_name'])
                            pose_x_his = pose_json[j]['pose']['position']['x']
                            pose_y_his = pose_json[j]['pose']['position']['y']
                            pose_z_his = pose_json[j]['pose']['position']['z']

                            pose_qw_his = pose_json[j]['pose']['orientation']['qw']
                            pose_qx_his = pose_json[j]['pose']['orientation']['qx']
                            pose_qy_his = pose_json[j]['pose']['orientation']['qy']
                            pose_qz_his = pose_json[j]['pose']['orientation']['qz']
                            upper_bound = abs(time_stamp_sample_his - time_stamp_pose)
                    pose_mat_his = np.eye(4, dtype = np.float64)
                    pose_mat_his[:3,:3] =Rot.from_quat([pose_qx_his, pose_qy_his, pose_qz_his, pose_qw_his]).as_matrix()
                    pose_mat_his[:3, 3] = [pose_x_his, pose_y_his, pose_z_his]
                    sensor_pose_mat_his = pose_mat_his @ ins_ex['lidar2imu']

                    pose_mat_cur = np.eye(4, dtype=np.float64)
                    pose_mat_cur[:3, :3] = Rot.from_quat(
                        [pose_qx, pose_qy, pose_qz, pose_qw]).as_matrix()
                    pose_mat_cur[:3, 3] = [pose_x, pose_y, pose_z]

                    pose_mat_cur = pose_mat_cur @ ins_ex['lidar2imu']

                    ego_his_trajs[index_his] = pose_mat_his[:3, 3]
                    ######Get Next Pose#####
                    #If has next frame#
                    if index_i + index_his - his_ts + 1 <=  len(sample_list) -1:
                        time_stamp_sample_next = int(sample_list[index_i + index_his -his_ts + 1].split('sample_')[-1])
                        # time_stamp_sample = int(g['frames'][i]['frame_name'].split('sample_')[-1])
                        upper_bound = 10000000000000000
                        for j in range(len(pose_json)):
                            # print(pose_json[j]['frame_name'])
                            # time_stamp_pose = int(pose_json[j]['frame_name'].split('sample_')[-1])
                            # print("Pose TimeStamp")
                            # print(pose_json[j]['timestamp'])
                            time_stamp_pose = 1000000 * float(pose_json[j]['timestamp'])
                            # time_stamp_pose = pow(10, (16-digit_length)) * float(pose_json[j]['timestamp'])
                            # if pose_json[j]['frame_name']==sample:
                            if abs(time_stamp_sample_next - time_stamp_pose) < upper_bound:
                                # print(pose_json[j]['frame_name'])
                                pose_x_next = pose_json[j]['pose']['position']['x']
                                pose_y_next = pose_json[j]['pose']['position']['y']
                                pose_z_next = pose_json[j]['pose']['position']['z']

                                pose_qw_next = pose_json[j]['pose']['orientation']['qw']
                                pose_qx_next = pose_json[j]['pose']['orientation']['qx']
                                pose_qy_next= pose_json[j]['pose']['orientation']['qy']
                                pose_qz_next = pose_json[j]['pose']['orientation']['qz']
                                upper_bound = abs(time_stamp_sample_next - time_stamp_pose)

                        pose_mat_next = np.eye(4, dtype=np.float64)
                        pose_mat_next[:3, :3] = Rot.from_quat(
                            [pose_qx_next, pose_qy_next, pose_qz_next, pose_qw_next]).as_matrix()
                        pose_mat_next[:3, 3] = [pose_x_next, pose_y_next, pose_z_next]

                        pose_mat_next= pose_mat_next @ ins_ex['lidar2imu']
                        ego_his_trajs_diff[index_his] = pose_mat_next[:3, 3] - ego_his_trajs[index_his]
                else:
                        ego_his_trajs[index_his] = ego_his_trajs[index_his + 1] - ego_his_trajs_diff[index_his + 1]
                        ego_his_trajs_diff[index_his] = ego_his_trajs_diff[index_his + 1]

            ###########global to ego at lcf######### global to ego at lcf
            ego_his_trajs = ego_his_trajs - np.array([pose_x, pose_y, pose_z])
            rot_mat = Quaternion([pose_qw, pose_qx, pose_qy, pose_qz]).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T

            # ego to lidar at lcf
            ego_his_trajs = ego_his_trajs - np.array(lidar2ego_trans)
            rot_mat = Quaternion(lidar2ego_rots).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]

            # get ego futute traj (offset format)
            ego_fut_trajs = np.zeros((fut_ts + 1, 3))
            ego_fut_masks = np.zeros((fut_ts + 1))
            for index_fut in range(fut_ts+1):

                ####If Cur Sample Exists##
                if index_i + index_fut <= len(sample_list)-1:

                    #######Get Pose#######
                    time_stamp_sample_fut_cur = int(sample_list[index_i + index_fut].split('sample_')[-1])
                    # time_stamp_sample = int(g['frames'][i]['frame_name'].split('sample_')[-1])
                    upper_bound = 10000000000000000
                    for j in range(len(pose_json)):
                        # print(pose_json[j]['frame_name'])
                        # time_stamp_pose = int(pose_json[j]['frame_name'].split('sample_')[-1])
                        # print("Pose TimeStamp")
                        # print(pose_json[j]['timestamp'])
                        time_stamp_pose = 1000000 * float(pose_json[j]['timestamp'])
                        # time_stamp_pose = pow(10, (16-digit_length)) * float(pose_json[j]['timestamp'])
                        # if pose_json[j]['frame_name']==sample:
                        if abs(time_stamp_sample_fut_cur - time_stamp_pose) < upper_bound:
                            # print(pose_json[j]['frame_name'])
                            pose_x_fut_cur = pose_json[j]['pose']['position']['x']
                            pose_y_fut_cur = pose_json[j]['pose']['position']['y']
                            pose_z_fut_cur = pose_json[j]['pose']['position']['z']

                            pose_qw_fut_cur = pose_json[j]['pose']['orientation']['qw']
                            pose_qx_fut_cur = pose_json[j]['pose']['orientation']['qx']
                            pose_qy_fut_cur = pose_json[j]['pose']['orientation']['qy']
                            pose_qz_fut_cur = pose_json[j]['pose']['orientation']['qz']
                            upper_bound = abs(time_stamp_sample_fut_cur - time_stamp_pose)

                    pose_mat_fut_cur  = np.eye(4, dtype=np.float64)
                    pose_mat_fut_cur[:3, :3] = Rot.from_quat(
                        [pose_qx_fut_cur, pose_qy_fut_cur, pose_qz_fut_cur, pose_qw_fut_cur]).as_matrix()
                    pose_mat_fut_cur[:3, 3] = [pose_x_fut_cur, pose_y_fut_cur, pose_z_fut_cur]

                    sensor_pose_mat_fut_cur = pose_mat_fut_cur @ ins_ex['lidar2imu']
                    ego_fut_trajs[index_fut] = sensor_pose_mat_fut_cur[:3, 3]
                    ego_fut_masks[index_fut] = 1

                if index_i + index_fut>= len(sample_list)-1:
                    ego_fut_trajs[index_fut + 1:] = ego_fut_trajs[index_fut]

            #########ego to lidar at lcf####
            ego_fut_trajs = ego_fut_trajs - np.array([pose_x, pose_y, pose_z])
            rot_mat = Quaternion([pose_qw, pose_qx
                                  , pose_qy, pose_qz]).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # ego to lidar at lcf
            ego_fut_trajs = ego_fut_trajs - np.array(lidar2ego_trans)
            rot_mat = Quaternion(lidar2ego_rots).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # drive command according to final fut step offset from lcf
            #if ego_fut_trajs[-1][0] >= 2:
            if ego_fut_trajs[-1][1] <= -2:
                command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs[-1][1] >= 2:
                command = np.array([0, 1, 0])  # Turn Left
            else:
                command = np.array([0, 0, 1])  # Go Straight
            # offset from lcf -> per-step offset
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]
            ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度
            ego_lcf_feat = np.zeros(9)
            # 根据odom推算自车速度及加速度
            ego_yaw,_,_ = quaternion_to_euler([pose_qw, pose_qx, pose_qy, pose_qz])

            ######Get Prev_pose###
            ego_yaw_prev = ''
            if  index_i >=1 :
                time_stamp_sample_prev_pose = int(sample_list[index_i - 1].split('sample_')[-1])
                # time_stamp_sample = int(g['frames'][i]['frame_name'].split('sample_')[-1])
                upper_bound = 10000000000000000
                for j in range(len(pose_json)):
                    # print(pose_json[j]['frame_name'])
                    # time_stamp_pose = int(pose_json[j]['frame_name'].split('sample_')[-1])
                    # print("Pose TimeStamp")
                    # print(pose_json[j]['timestamp'])
                    time_stamp_pose = 1000000 * float(pose_json[j]['timestamp'])
                    # time_stamp_pose = pow(10, (16-digit_length)) * float(pose_json[j]['timestamp'])
                    # if pose_json[j]['frame_name']==sample:
                    if abs(time_stamp_sample_prev_pose - time_stamp_pose) < upper_bound:
                        # print(pose_json[j]['frame_name'])
                        pose_x_prev_pose = pose_json[j]['pose']['position']['x']
                        pose_y_prev_pose = pose_json[j]['pose']['position']['y']
                        pose_z_prev_pose = pose_json[j]['pose']['position']['z']

                        pose_qw_prev_pose = pose_json[j]['pose']['orientation']['qw']
                        pose_qx_prev_pose = pose_json[j]['pose']['orientation']['qx']
                        pose_qy_prev_pose = pose_json[j]['pose']['orientation']['qy']
                        pose_qz_prev_pose = pose_json[j]['pose']['orientation']['qz']
                        upper_bound = abs(time_stamp_sample_prev_pose - time_stamp_pose)
                ego_yaw_prev, _, _ = quaternion_to_euler([pose_qw_prev_pose, pose_qx_prev_pose, pose_qy_prev_pose,
                                                         pose_qz_prev_pose])

            ######Get  Next  pose###
            ego_yaw_next = ''
            ####################
            if index_i +1  <= len(sample_list)-1:
                time_stamp_sample_next_pose = int(sample_list[index_i + 1].split('sample_')[-1])
                # time_stamp_sample = int(g['frames'][i]['frame_name'].split('sample_')[-1])
                upper_bound = 10000000000000000
                for j in range(len(pose_json)):
                    # print(pose_json[j]['frame_name'])
                    # time_stamp_pose = int(pose_json[j]['frame_name'].split('sample_')[-1])
                    # print("Pose TimeStamp")
                    # print(pose_json[j]['timestamp'])
                    time_stamp_pose = 1000000 * float(pose_json[j]['timestamp'])
                    # time_stamp_pose = pow(10, (16-digit_length)) * float(pose_json[j]['timestamp'])
                    # if pose_json[j]['frame_name']==sample:
                    if abs(time_stamp_sample_next_pose - time_stamp_pose) < upper_bound:
                        # print(pose_json[j]['frame_name'])
                        pose_x_next_pose = pose_json[j]['pose']['position']['x']
                        pose_y_next_pose = pose_json[j]['pose']['position']['y']
                        pose_z_next_pose = pose_json[j]['pose']['position']['z']

                        pose_qw_next_pose = pose_json[j]['pose']['orientation']['qw']
                        pose_qx_next_pose = pose_json[j]['pose']['orientation']['qx']
                        pose_qy_next_pose = pose_json[j]['pose']['orientation']['qy']
                        pose_qz_next_pose = pose_json[j]['pose']['orientation']['qz']
                        upper_bound = abs(time_stamp_sample_next_pose - time_stamp_pose)

                ego_yaw_next, _, _ = quaternion_to_euler([pose_qw_next_pose, pose_qx_next_pose, pose_qy_next_pose,
                                                           pose_qz_next_pose])
            if ego_yaw_prev != '':
                ego_w = (ego_yaw - ego_yaw_prev) / 0.1
                ego_v = np.linalg.norm((np.array([pose_x, pose_y])- np.array([pose_x_prev_pose, pose_y_prev_pose]))) / 0.1
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw ), ego_v * math.sin(ego_yaw )
            else:
                ego_w = (ego_yaw_next - ego_yaw) / 0.1
                ego_v = np.linalg.norm((np.array([pose_x_next_pose, pose_y_next_pose]) - np.array([pose_x, pose_y]))) / 0.1
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw), ego_v * math.sin(ego_yaw )

            delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
            delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
            v0 = np.sqrt(delta_x ** 2 + delta_y ** 2)

            ego_lcf_feat[:2] = np.array([ego_vx, ego_vy])  # can_bus[13:15]

            ego_lcf_feat[4] = ego_w  # can_bus[12]
            ego_length = 5.0
            ego_width = 3.0
            ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
            ego_lcf_feat[7] = v0
            #####Steer#####
            ego_lcf_feat[8] = ego_w *10

            imu_datas = mmcv.load(os.path.join(clips_path, clip, 'imu.json' ))
            upper_bound_imu = 10000000000000000
            time_stamp_sample_reference= int(sample_list[index_i].split('sample_')[-1])
            for data in imu_datas:
                time_stamp_imu = float(data['timestamp']) * 1000000
                if abs(time_stamp_sample_reference - time_stamp_imu) < upper_bound_imu:
                    ax = data['imu']['linear_acceleration']['y']
                    ay = - data['imu']['linear_acceleration']['x']
                    omega = data['imu']['euler_angles']['z']
                    upper_bound = abs(time_stamp_sample_reference - time_stamp_imu)

            ego_lcf_feat[2] = ax
            ego_lcf_feat[3] = ay
            ego_lcf_feat[4] = omega

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = gt_names
            info['gt_velocity'] = gt_velocity.reshape(-1, 2)
            #info['num_lidar_pts'] = np.array(
            #    [a['num_lidar_pts'] for a in annotations])
            #info['num_radar_pts'] = np.array(
            #    [a['num_radar_pts'] for a in annotations])
            #info['valid_flag'] = valid_flag
            info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts * 2).astype(np.float32)
            info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
            info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
            info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
            info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
            info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd'] = command.astype(np.float32)
            info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)
            infos.append(info)
        metadata = {}
        #data_pkl = dict(infos=infos, metadata=metadata)
        mmcv.dump(infos, os.path.join(clips_path, 'annos_1217',  clip +'_vad_infos_1214.json'))
if __name__ == "__main__":
    main()















