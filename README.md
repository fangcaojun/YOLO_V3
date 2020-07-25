# YOLO_V3
实现步骤参考CSDN：https://blog.csdn.net/weixin_44318872/article/details/104660836
LabelImg下载链接为：https://github.com/fangcaojun/ImageProcess/tree/master/labelImg-master



**1.从头训练命令：**
#--batch_size默认为2，可以根据自己电脑配置进行更改，值越大耗费的GPU显存越多
```python
python3 train.py --batch_size 2  --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/yolov3.weights
```

**2.从训练终止处继续训练命令：**
#需修改checkpoints/yolov3_ckpt_2.pth为自己的值（在checkpoint文件夹中可以找到）
```python
python3 train.py --batch_size 2  --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights checkpoints/yolov3_ckpt_2.pth
```

**3.图片检测**
#修改image_folder的值为待检测图片文件或文件夹位置，weight_path改成自己的checkpoints值
```python
python3 detect.py --image_folder data/samples/ --weights_path checkpoints/yolov3_ckpt_6.pth --model_def config/yolov3-custom.cfg --class_path data/classes.names
```

**4.实时摄像头检测**
```
python3 realtime_detection.py
```

**5.tensorboard查看**
```python
tensorboard --logdir='logs' --port=6006
```


