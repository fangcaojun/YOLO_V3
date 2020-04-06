import os
import random
import pickle
from os.path import join
from os import listdir, getcwd
import xml.etree.ElementTree as ET

#测试集test占总数据集比例=（1-trainval_percent）,这里是0.2，可根据需要更改trainval_percent的值
trainval_percent = 0.8
#训练集train占trainval（包含训练集train和验证集val）中的比例
train_percent = 0.7

#xml文件的路径
xmlfilepath = 'VOCdevkit2007/Annotations/'
#转换生成的txt文件保存目录
txtsavepath = 'VOCdevkit2007/ImageSets/Main/'

total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

#生成的数据集保存位置
ftrainval = open('VOCdevkit2007/ImageSets/trainval.txt', 'w')
ftest = open('VOCdevkit2007/ImageSets/test.txt', 'w')
ftrain = open('VOCdevkit2007/ImageSets/train.txt', 'w')
fval = open('VOCdevkit2007/ImageSets/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()


sets = ['train', 'test','val']

#改成自己标注的标签名称
classes = ["spot"]

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def convert_annotation(image_id):
    in_file = open('VOCdevkit2007/Annotations/%s.xml' % (image_id))
    out_file = open('VOCdevkit2007/ImageSets/Main/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('VOCdevkit2007/ImageSets/Main/'):
        os.makedirs('VOCdevkit2007/ImageSets/Main/')
    #训练集train、测试集test、验证集val所在位置
    image_ids = open('VOCdevkit2007/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('VOCdevkit2007/ImageSets/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        #图片所在位置
        list_file.write('data/VOCdevkit2007/JPEGImages/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()