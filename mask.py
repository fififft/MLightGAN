import torch
import cv2
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageOps
import torchvision.transforms as transforms
def auto_illu_mask(input_op,illu_min,illu_max,p,q):
       # input_op = Image.open("C:/Users/0/Desktop/新建文件夹 (3)/2.png")
        input_op = to_tensor(input_op)
        illu_k = input_op.clamp(min=1e-4)
        illu_masks = []
        scales = []
        b, _, _ = illu_k.shape
        for i in range(b):
            i_k = illu_k[i]
            # k = 3.141592653589793 / (i_max - i_min)
            # illu_mask = 0.5 * (torch.cos(k * (i_k - i_min)) + 1)
            key_point = illu_min * (i_k.median() - i_k.min())
            la = i_k.min() + key_point
            key_point = illu_max * (i_k.max() - i_k.median())
            lb = i_k.max() - key_point
            if lb < 0.92:
                lb = 0.92
            else:
                lb = lb
            illu_mask_mid = (i_k > la) * (i_k < lb)
            illu_mask_low = (1 / (1 + (i_k < la) * (i_k - la) * (i_k - la) * (p ** 2))) * (i_k < la)
            illu_mask_high = (1 / (1 + (i_k > lb) * (i_k - lb) * (i_k - lb) * (q ** 2))) * (i_k > lb)
            illu_mask = illu_mask_mid + illu_mask_low + illu_mask_high
            illu_masks.append(illu_mask)
            k = 1 / (lb - la)
            scales.append(k)
        i_mask = torch.stack(illu_masks)
        k = torch.stack(scales).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mask_image = i_mask * illu_k
        #return i_mask, k
        return mask_image

        #input_op = Image.open("C:/Users/0/Desktop/mask/2.png")
        #mask_image = auto_illu_mask(input_op,0.1,0.9,5,5)
        #print(mask_image.size())
        #mask_image_np = np.transpose(mask_image.squeeze().numpy(), (1, 2, 0))
        #fig, ax = plt.subplots()
        #ax.imshow(mask_image_np)
        #plt.show()
if __name__ == "__main__":
    # 加载数据和计算掩码和缩放张量
    image = Image.open('C:/Users/0/Desktop/mask/12.jpg')
    mask_image = auto_illu_mask(image, 0.01, 0.9, 5, 5)
    # 创建保存图片的文件夹
    save_folder = './yuan'
    os.makedirs(save_folder, exist_ok=True)

    # 将mask_image和k保存为图像文件
    mask_image_pil = to_pil_image(mask_image.squeeze())
    mask_image_path = os.path.join(save_folder, '14.png')
    mask_image_pil.save(mask_image_path)
    # 打印保存的图片路径
    print("图片已保存在以下路径：")
    print(f"mask_image4.png -> {mask_image_path}")
    #print(f"k_image.png -> {k_image_path}")
