# ML lab3 README

将数据放到data/目录下，并且重命名为video_data/

运行如下命令，开始训练:

```shell
# 预训练cnn
python main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 4 --sample_size 150 --lr_rate 1e-4 --n_classes 10 --model cnnlstm
# 自定义cnn
python main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 4 --sample_size 150 --lr_rate 1e-4 --n_classes 10 --model custom_cnnlstm3
```

运行如下命令，进行预测分类推理:

```shell
python inference.py --model <模型种类> --n_classes 10 --resume_path <模型路径> --predict_video_path <预测的视频的路径>

eg:
python inference.py --model cnnlstm --n_classes 10 --resume_path snapshots/cnnlstm-Epoch-100-Loss-0.3323290095478296.pth --predict_video_path data/video_data/SumoWrestling/v_SumoWrestling_g01_c01.avi
```

