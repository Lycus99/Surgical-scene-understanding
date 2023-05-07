## Deep Long-Tailed Learning: A Survey

1. 长尾分布的问题：the trained model can be easily biased towards head classes with massive training data, leading to poor performance on tail classes that have limited data. 

2. 常见的解决方法：
![image](https://user-images.githubusercontent.com/109274751/236391929-f31f4df1-74aa-48fa-b6c4-4ff21fddc620.png)

imbalance ratio: $n_1/n_K$

3. Class Re-balancing

3.1 Re-sampling
1. random over-sampling: randomly repeats the tail samples 过拟合尾部
2. random under-sampling: randomly discards the head sample 欠拟合头部
3. class-balanced re-sampling: 
  1. Decoupling[32](*):
    1. instance-balanced: each sample has an equal probability of being sampled
    2. class-balanced: each class has an equal probability
    3. square-root: probability for each class is related to the square root of sample size
    4. Progressively-balanced: instance-balanced --> class-balanced sampling.
    结论：将模型解耦为backbone和classifier，训练backbone时使用instance-balanced，之后冻住。训练classifier时使用re-sampling strategy。两阶段方法。
    发现了一个方法叫做De-confound-TDE
    先用Swin达到SelfD的baseline，之后再尝试加别的（Multi-task，long-tail）
  2. Simple Calibration:
    bi-level class-balanced sampling
  3. Dynamic curriculum learning: curriculum strategy
