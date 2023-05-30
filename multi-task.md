## Multi-Task learning

arxiv: 2004.13379   Multi-Task Learning for Dense Prediction Tasks: A Survey

# 1. Encoder-Focused 

# 2. Decoder-Focused

1. PAD-Net

![image](https://github.com/Lycus99/Surgical-scene-understanding/assets/109274751/913cca9c-6b0f-4649-a88f-e274b6e3ccf8)

利用已经解耦的任务特征。第k个任务的输出等于第k个任务的输入加上其他任务与第k个任务的attention

![image](https://github.com/Lycus99/Surgical-scene-understanding/assets/109274751/b0d54321-e3ea-4de2-be60-c0a88a3bbcf4)


2. PAP-Net

统计不同任务的亲和度。GFLOPS很大


# 3. Task Interference 

1. Loss/Gradient balancing

通过控制每个任务对总任务的梯度或损失的贡献

同方差不确定性：Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. arXiv2017-05-19

动态调整不同梯度的大小GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks  
arXiv: Computer Vision and Pattern Recognition2017-11-07

多目标优化：Multi-Task Learning as Multi-Objective Optimization arXiv: Learning2018-10-10

2. Parameter partitioning

自适应的关注特征图的不同区域

在filter-level使用attention，是每个任务在每一层选择一部分参数
End-to-End Multi-Task Learning with Attention. Cornell University - arXiv2018-03-28

task-specific squeeze-and-excitation module (channel attention). soft 参数划分
Attentive Single-Tasking of Multiple Tasks. arXiv: Computer Vision and Pattern Recognition2019-04-18

task routing module. 随机给每个任务分配一个子网络
Many Task Learning with Task Routing. arXiv: Computer Vision and Pattern Recognition2019-03-28

在训练中自适应更新参数划分. Maximum Roaming Multi-Task Learning. 
AAAI Conference on Artificial Intelligence2020-06-17


3. Architectural design
