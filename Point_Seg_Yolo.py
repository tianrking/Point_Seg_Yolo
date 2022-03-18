from final_integration import Myinput
import os
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image

input_filepath = "/storage/lol/gg/yolov5/Point_Seg_data_1/"
output_filepath = "/storage/lol/gg/yolov5/Point_Seg_data_Output/"
output_k = len(os.listdir(output_filepath)) + 1  
output_filepath = output_filepath + str(output_k) + "/"
# print(output_filepath)

class b(object):
    def __init__(self,pic1,pic2,pic3,pic4,output_dir=1):
        self.pic1 = pic1
        self.pic2 = pic2
        self.pic3 = pic4
        self.pic4 = pic3
        self.output_dir = output_dir

def merge(GG):

    input_filepath = GG.pic1
    all_fusion_folder_path = GG.pic2
    single_fusion_folder_path = GG.pic3
    output_dir = GG.output_dir

    im_list = [GG.pic1, GG.pic2,  GG.pic3, GG.pic4]
    im_list =  [Image.open(fn) for fn in im_list]

    ims = []

    for i in im_list:
        new_img = i.resize((1280, 1280), Image.BILINEAR)
        ims.append(new_img)

    width, height = ims[0].size

    result = Image.new(ims[0].mode, (height * len(ims),width))

    for i, im in enumerate(ims):
        result.paste(im, box=(i * height, 0))
    
    result.save(output_dir)



k=0
import os
from os.path import isfile, join
for i in range(0,len(os.listdir(input_filepath+"0_person_image"))):
    pass
for gg in (os.listdir(input_filepath+"0_person_image")):
    i = gg.split(".png")[0]

    # image_path = input_filepath + "0_images/%s.jpg"%str(i)
    # image_yolo_path = input_filepath + "exp6_image_data/%s.jpg"%str(i)

    # bin_path = input_filepath + "exp6_bin_data/%s.bin"%str(k)
    # # bin_path = input_filepath + "exp6_bin_data/"

    # calib_text_path_final = input_filepath + "calib/1.txt"
    # position_path = input_filepath + "poisition_data/%s.txt"%str(k)


    image_path = input_filepath + "0_person_image/%s.png"%str(i)
    image_yolo_path = input_filepath + "4_detect_image/%s.png"%str(i)
    bin_path = input_filepath + "1_person_bin/%s.bin"%str(i)
    calib_text_path_final = input_filepath + "2_calibtxt/%s.txt"%str(i)
    position_path = input_filepath + "3_poisition_data/%s.txt"%str(i)

    # print(position_path)
    # continue

    seg_bin_folder_path = output_filepath + "seg_bin/"
    single_fusion_folder_path = output_filepath + "single_fusion/"
    all_fusion_folder_path = output_filepath + "all_fusion/"
    all_in_one_path = output_filepath + "all_in_one/"

    try:
        os.makedirs(seg_bin_folder_path)
        os.makedirs(single_fusion_folder_path)
        os.makedirs(all_fusion_folder_path)
        os.makedirs(all_in_one_path)
    except:
        input_message = Myinput()
        input_message.data_input(image_path, bin_path, calib_text_path_final, position_path)
        input_message.data_output(seg_bin_folder_path,single_fusion_folder_path,all_fusion_folder_path)
        input_message.seg_point()

        # GG=b(
        #     image_path,
        #     image_yolo_path,
        #     single_fusion_folder_path + "%s.jpg"%str(i),
        #     all_fusion_folder_path + "%s.jpg"%str(i),
        #     all_in_one_path + "%s.jpg"%str(i)
        #     )


        GG=b(
            image_path,
            image_yolo_path,
            single_fusion_folder_path + "%s.png"%str(i),
            all_fusion_folder_path + "%s.png"%str(i),
            all_in_one_path + "%s.png"%str(i)
            )
            
        merge(GG)

        k = k + 1


    print(i)

print("success")
