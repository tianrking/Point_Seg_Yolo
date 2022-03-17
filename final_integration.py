# -*- coding:utf-8 -*-
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import math
import os
# import open3d
# import struct
# import mayavi.mlab

# """
# 实际数据集测试

def lidar_cam_all_fusion(
    img_path,
    binary_path,
    calib_text_path,
    fusion_path
    ):
    sn = int(sys.argv[1]) if len(sys.argv) > 1 else 7  # default 0-7517
    name = '%06d' % sn  # 6 digit zeropadding
    img = img_path
    file_name = os.path.basename(img)
    file_name_forward = file_name.split('.')[0]
    # print(file_name_forward)
    # print(file_name)
    binary = binary_path
    # with open(f'./testing/calib1/{name}.txt','r') as f:
    #     calib1 = f.readlines()
    f = open(calib_text_path, 'r')
    calib1 = f.readlines()
    P0 = np.matrix([float(x) for x in calib1[0].strip('\n').split(' ')[1:]]).reshape(3, 4)

    Tr_velo_to_cam1 = np.matrix([float(x) for x in calib1[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam1 = np.insert(Tr_velo_to_cam1, 3, values=[0, 0, 0, 1], axis=0)

    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points, 3, 1, axis=1).T
    cam = P0 * Tr_velo_to_cam1 * velo  # P2 * R0_rect * Tr_velo_to_cam * velo
    #####################################
    # get u,v,z
    cam[:2] /= cam[2, :]
    # do projection staff
    fig, ax1 = plt.subplots(1)
    png = mpimg.imread(img)
    IMG_H, IMG_W, _ = png.shape
    # restrict canvas in range
    plt.axis([0, IMG_W, IMG_H, 0])
    plt.xticks([])  # 不显示x轴
    plt.yticks([])  # 不显示y轴
    plt.imshow(png)
    # filter point out of canvas
    u, v, z = cam
    ############绘制点云投影图像##################
    # print("u")
    # print(u)
    # print("v")
    # print(v)
    # print("z")
    # print(z)
    plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)

    # plt.title(file_name)
    ax1.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(30, 10)
    # manager = plt.get_current_fig_manager()
    # manager.Maximize(True)
    # manager.window.showMaximized()
    # plt.show()
    seg_fusion_image = fusion_path + '/' + file_name
    fig.savefig(seg_fusion_image, dpi=300, transparent=True, pad_inches=0, bbox_inches='tight')
    plt.cla()
    plt.close("all")
    return seg_fusion_image

# 函数输入：参数意义：图片路径，点云路径，标定文件路径，目标检测位置文件，融合文件夹存储路径（默认）；输出：返回融合文件的路径
def lidar_cam_fusion(
    img_path,
    binary_path,
    calib_text_path,
    position_path,
    distance,
    fusion_path
    ):
    fi = open(position_path, 'r')
    lines = fi.readlines()
    coordinate_data = []
    for line in lines:
        line = line.strip('\n').split(' ')[0:-1]
        line_new = []
        for ls in line:
            line_new.append(eval(ls))
        coordinate_data.append(line_new)
    fi.close()
    sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
    name = '%06d'%sn # 6 digit zeropadding
    img = img_path
    file_name = os.path.basename(img)
    file_name_forward = file_name.split('.')[0]
    # print(file_name_forward)
    # print(file_name)
    binary = binary_path
    # with open(f'./testing/calib1/{name}.txt','r') as f:
    #     calib1 = f.readlines()
    f = open(calib_text_path,'r')
    calib1 = f.readlines()
    P0 = np.matrix([float(x) for x in calib1[0].strip('\n').split(' ')[1:]]).reshape(3,4)

    Tr_velo_to_cam1 = np.matrix([float(x) for x in calib1[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam1 = np.insert(Tr_velo_to_cam1,3,values=[0,0,0,1],axis=0)

    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
    points = scan[:, 0:3] # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points,3,1,axis=1).T
    cam = P0 * Tr_velo_to_cam1 * velo #P2 * R0_rect * Tr_velo_to_cam * velo
    #####################################
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    fig, ax1 = plt.subplots(1)
    png = mpimg.imread(img)
    IMG_H,IMG_W,_ = png.shape
    # restrict canvas in range
    plt.axis([0,IMG_W,IMG_H,0])
    plt.xticks([])  # 不显示x轴
    plt.yticks([])  # 不显示y轴
    plt.imshow(png)
    # filter point out of canvas
    u,v,z = cam
   ############绘制点云投影图像##################
    # print("u")
    # print(u)
    # print("v")
    # print(v)
    # print("z")
    # print(z)
    plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
    for i in range(len(coordinate_data)):
        plt.text(coordinate_data[i][1],(coordinate_data[i][2]+coordinate_data[i][3])//2,distance[i],fontsize = 20,color="yellow")
    #plt.title(file_name)
    ax1.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(30,10)
    # manager = plt.get_current_fig_manager()
    # manager.Maximize(True)
    # manager.window.showMaximized()
    # plt.show()
    seg_fusion_image = fusion_path + '/' + file_name
    fig.savefig(seg_fusion_image, dpi=300, transparent=True, pad_inches=0, bbox_inches='tight')
    plt.cla()
    plt.close("all")
    return seg_fusion_image
    # plt.pause(1)
    # plt.close('all')


"""
    指定框的坐标，分割框内点云并画框
"""
def count_most(list,min,max,interval):
    count_max = 0
    result_min =0
    while min <= max:
        i = 0
        for elements in list:
            if (elements > min) & (elements < (min + interval)):
                i += 1
        if (i > count_max):
            count_max = i
            result_min = min + interval
  
        min = min + interval

    # print(result_min)
    return result_min
# 输入：参数意义：图片文件路径，点云文件路径，标定文件路径，检测框位置文件，分割点云文件夹路径（默认）；输出：返回目标的距离列表，返回点云分割之后和图像融合的数据
def seg_points_and_fusion(
    img_path,
    binary_path,
    calib_text_path,
    position_path,

    seg_path, ##seg_bin_folder_path,
    single_fusion_folder_path,
    all_fusion_folder_path
    ):
    fi = open(position_path, 'r')
    lines = fi.readlines()
    coordinate_data = []
    for line in lines:
        line = line.strip('\n').split(' ')[0:-1]
        line_new = []
        for ls in line:
            line_new.append(eval(ls))
        coordinate_data.append(line_new)
    n_m = len(coordinate_data)
    fi.close()
    object_distance = []
    # sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
    # name = '%06d'%sn # 6 digit zeropadding
    img = img_path
    file_name = os.path.basename(img)
    file_name_forward = file_name.split('.')[0]
    # print(file_name)
    binary = binary_path
    # with open(f'./testing/calib1/{name}.txt','r') as f:
    #     calib1 = f.readlines()
    f = open(calib_text_path, 'r')
    calib1 = f.readlines()

    P0 = np.matrix([float(x) for x in calib1[0].strip('\n').split(' ')[1:]]).reshape(3,4)

    Tr_velo_to_cam1 = np.matrix([float(x) for x in calib1[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam1 = np.insert(Tr_velo_to_cam1,3,values=[0,0,0,1],axis=0)

    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
    scan_T = scan.T
    points = scan[:, 0:3] # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points,3,1,axis=1).T
    x_velo = velo[0].flatten()
    y_velo = velo[1].flatten()
    z_velo = velo[2].flatten()
    d_velo = []
    for i in range(len(x_velo)):
        d_velo_data = math.sqrt((x_velo[i] ** 2 + y_velo[i] ** 2 + z_velo[i] ** 2))
        d_velo.append(d_velo_data)
    o_velo = scan_T[3].flatten()

    cam = P0 * Tr_velo_to_cam1 * velo #P2 * R0_rect * Tr_velo_to_cam * velo
    # print("cam:")
    # print(cam)
    # print(cam.shape)
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    fig, ax1 = plt.subplots(1)
    png = mpimg.imread(img)
    IMG_H,IMG_W,_ = png.shape
    # restrict canvas in range
    plt.axis([0,IMG_W,IMG_H,0])
    plt.xticks([])  # 不显示x轴
    plt.yticks([])  # 不显示y轴
    plt.imshow(png)
    # filter point out of canvas
    # u,v,z = cam
    # generate color map from depth
    u,v,z = cam
    ###############画框#############
    # u_max = 643
    # print(u_max)
    # u_min = 608
    # print(u_min)
    # v_max = 208
    # print(v_max)
    # v_min = 182
    # print(v_min)

    ######提取框内点云#######
    '''矩阵转数组并转化为一维数组'''
    x_cam = cam[0].getA().flatten()
    y_cam = cam[1].getA().flatten()
    z_cam = cam[2].getA().flatten()
    # cam_x_test=[]
    # cam_y_test=[]

    # cam_z_test=[]
    velo_x_test = []
    velo_y_test = []
    velo_z_first_test = []
    velo_z_test = []
    velo_o_test = []
    velo_d_test = []
    rect = []
    for i in range(n_m):
        cam_z_test = []
        distance_test = []
        for index,x_elements in enumerate(x_cam):
            if (x_elements < coordinate_data[i][0]) & (x_elements > coordinate_data[i][1]):
                if (y_cam[index] < coordinate_data[i][2]) & (y_cam[index] > coordinate_data[i][3]):
                    distance_test.append((d_velo[index]))

        if len(distance_test) == 0:
            distance_test = [-1]
        velo_z_threshold = count_most(distance_test, int(min(distance_test)) - 1, int(max(distance_test)) + 1, 1)
        # print(velo_z_threshold)
        left_first_threshold = velo_z_threshold + 1
        # print("left")
        # print(left_first_threshold)
        right_first_threshold = velo_z_threshold - 1
        # print("right")
        # print(right_first_threshold)
        for index,x_elements in enumerate(x_cam):
            if (x_elements < coordinate_data[i][0]) & (x_elements > coordinate_data[i][1]):
                if (y_cam[index] < coordinate_data[i][2]) & (y_cam[index] > coordinate_data[i][3]):
                    # if ((z_cam[index]) < cam_threshold) & ((z_cam[index]) > cam_threshold - 1):
                    if ((d_velo[index]) < left_first_threshold) & ((d_velo[index]) > right_first_threshold) & (x_velo[index]>0):
                    # if ((z_velo[index]) < left_first_threshold) & ((z_velo[index]) > right_first_threshold):
                        velo_x_test.append(x_velo[index])
                        velo_y_test.append(y_velo[index])
                        velo_z_test.append(z_velo[index])
                        velo_o_test.append(o_velo[index])
                        velo_d_test.append(d_velo[index])
        d_sum = 0
        for d_point in velo_d_test:
            d_sum = d_point + d_sum
        n_test = len(velo_d_test)
        if n_test == 0:
            n_test = 1
        d_average = d_sum / n_test
        # print("第"+str(i)+"个目标的距离：{:.2f}".format(d_average))
        d_average = "{:.2f}".format(d_average)
        object_distance.append(d_average)
        velo_test_list = []
        velo_test_list.append(velo_x_test)
        velo_test_list.append(velo_y_test)
        velo_test_list.append(velo_z_test)
        velo_test_list.append(velo_o_test)
        velo2 = np.mat(np.array(velo_test_list))
        ##############写入bin文件##############
        a = velo2.T.reshape(1, -1)
        b = a.A
        c = b.flatten()
        # print(c)
        seg_bin_test_path = seg_path + '/'+file_name_forward+'.bin'
        file = open(seg_bin_test_path, 'wb')
        file.write(c)
        file.close()
   
    seg_fusion_image = lidar_cam_fusion(
        img_path, 
        seg_bin_test_path, 
        calib_text_path,
        position_path,
        object_distance,
        single_fusion_folder_path
        )

    all_fusion_image = lidar_cam_all_fusion(
        img_path, 
        binary_path, 
        calib_text_path,
        all_fusion_folder_path
        )

    return object_distance,seg_fusion_image,all_fusion_image

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)


def display(path):

    pcd=open3d.open3d.geometry.PointCloud()

    example=read_bin_velodyne(path)

    pcd.points= open3d.open3d.utility.Vector3dVector(example)
    open3d.open3d.visualization.draw_geometries([pcd])

# 参数意义：图片文件路径，点云文件路径，标定文件路径，检测框坐标[[xmax,xmin,ymax,ymin]...]，检测框的数量
class Myinput(object):
    def __init__(self):
        print("Hey")

    def data_input(self,image,point_bin,calib_text,position_path):
        self.input_image = image
        self.input_point_bin = point_bin
        self.input_calib_text = calib_text
        self.input_position_path = position_path
    
    def data_output(self,seg_bin_folder_path,single_fusion_folder_path,all_fusion_folder_path):
        self.output_seg_bin_folder_path = seg_bin_folder_path
        self.output_single_fusion_folder_path = single_fusion_folder_path
        self.output_all_fusion_folder_path = all_fusion_folder_path

    def seg_point(self):
        seg_points_and_fusion(
            self.input_image,
            self.input_point_bin,
            self.input_calib_text,
            self.input_position_path,

            self.output_seg_bin_folder_path,
            self.output_single_fusion_folder_path,
            self.output_all_fusion_folder_path,

            )

##### Uncomment to test ##########

# image_path = r'./0_image/40.jpg'
# bin_path = r'./1_point_bin/40.bin'
# calib_text_path_final = r'./2_calibtxt/000007.txt'
# position_path = r'./4_poisitiontxt/40.txt'
# coordinate_data_final = [[434,345,380,176],[328,267,331,188],[246,210,294,207]]

# seg_bin_folder_path = r'/storage/lol/kitti-velo2cam-master/5_seg_bin/'
# single_fusion_folder_path = r'/storage/lol/kitti-velo2cam-master/6_seg_fusion/'
# all_fusion_folder_path = r'/storage/lol/kitti-velo2cam-master/7_all_fusion/'
# input_message = Myinput()
# input_message.data_input(image_path, bin_path, calib_text_path_final, position_path)
# input_message.data_output(seg_bin_folder_path,single_fusion_folder_path,all_fusion_folder_path)
# input_message.seg_point()
# print("success")
