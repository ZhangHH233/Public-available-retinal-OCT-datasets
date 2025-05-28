import SimpleITK as sitk
import numpy as np
from PIL import Image
import os

from pathlib import Path

"""
将RETOUCH中的volume转换成一系列二维图像，保存为png。
"""



class RETOUCH_preprocess():
    def __init__(self):
        
        self.root_dir = r"F:\Data\Public\retouch"
        
        # self.train_volume, self.test_volume = self.get_dirs()
        
        self.save_dir = r"F:\Data"
        self.oct_list, self.ref_list= self.get_dirs()

    def get_dirs(self):
        
        oct =list( Path(self.root_dir).glob('*Cirrus/*/*/oct.mhd'))
        ref =list( Path(self.root_dir).glob('*Cirrus/*/*/reference.mhd'))
        
        # print(oct)
        
        return oct, ref
    
    
    def save_out(self, img_list, ref_list, save=False):
        for img_path, ref_path in zip(img_list, ref_list):
            # 读取图像和参考
            img = sitk.ReadImage(str(img_path))
            ref = sitk.ReadImage(str(ref_path))
            
            
            
            img_array = sitk.GetArrayFromImage(img)  # 获取 NumPy 数组
            ref_array = sitk.GetArrayFromImage(ref)  # 获取 NumPy 数组
            
            img_array = (img_array- np.min(img_array)) / (np.max(img_array) - np.min(img_array))  # 缩放到 [0, 1]
            img_array = (img_array * 255).astype(np.uint8)  # 缩放到 [0, 255]


            # 创建输出文件夹
            output_folder = self.save_dir
            

            # 遍历每个切片
            for i in range(img_array.shape[0]):
                img_slice = img_array[i, :, :]
                ref_slice = ref_array[i, :, :]  # 乘以 255

                # 替换文件名部分
                img_filename = str(img_path).replace(self.root_dir, self.save_dir)
                ref_filename = str(ref_path).replace(self.root_dir, self.save_dir)
                
                img_filename = img_filename.replace('.mhd','')
                ref_filename = ref_filename.replace('.mhd','')
                
                os.makedirs(img_filename, exist_ok=True)
                os.makedirs(ref_filename, exist_ok=True)

                # 保存切片
                img_file_path = os.path.join(img_filename, f'img_slice_{i}.png')
                ref_file_path = os.path.join(ref_filename, f'ref_slice_{i}.png')

                # # 转换为图像并保存
                if save:
                    sitk.WriteImage(sitk.GetImageFromArray(img_slice.astype(np.uint8)), img_file_path)
                    sitk.WriteImage(sitk.GetImageFromArray(ref_slice.astype(np.uint8)), ref_file_path)

        print("切片保存完成！")
    
    def load_data(self, volume_list, toSave=True):
    
        # 读取 MHD 文件
        image = sitk.ReadImage(r'F:\Data\Public\retouch\TrainingSet-Topcon\Topcon_part2\628be4afa549cfc60031b1450215618b\reference.mhd')
        image_array = sitk.GetArrayFromImage(image)  # 获取 NumPy 数组

        # 创建输出文件夹
        output_folder = r'F:\Data\Public\retouch\TrainingSet-Topcon\Topcon_part2\628be4afa549cfc60031b1450215618b\reference'
        os.makedirs(output_folder, exist_ok=True)

        # 遍历每个切片并保存为 PNG
        for i in range(image_array.shape[0]):  # 遍历 z 轴
            slice_image = image_array[i, :, :]  # 获取切片
            img = Image.fromarray(slice_image)  # 转换为图像（调整数据类型）
            img.save(os.path.join(output_folder, f'slice_{i}.png'))  # 保存为 PNG

        print("切片保存完成！")

oct, ref = RETOUCH_preprocess().get_dirs()

RETOUCH_preprocess().save_out(oct, ref)
