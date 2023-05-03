# Surgical-scene-understanding
arxiv上与手术场景理解的最新论文


## SurgicalGPT: End-to-End Language-Vision GPT for Visual Question Answering in Surgery
作者：Lalithkumar Seenivasan， Hongliang Ren。 NUS，CUHK

地址：https://arxiv.org/pdf/2304.09974.pdf

总结：
1. 使用LLM来进行手术领域的VQA，使用的模型是GPT2。
2. 数据集是在已有的手术数据集（EndoVis18，Cholec80，PSI-AVA）基础上，将他们扩展成VQA。数据集没有公开
3. 网络结构：
![image](https://user-images.githubusercontent.com/109274751/234734939-79cac6ce-3a53-4322-b644-dd61b0d8a7bb.png)
4. 效果：
![image](https://user-images.githubusercontent.com/109274751/234735009-2aaaba72-5b05-4f16-bf3e-970fb0b9b71f.png)
5. 由于数据集没有公开，不好跟别的方法比较。不过从效果来看已经很好了，这个组之前就做过手术领域的Image Captioning，感觉值得follow，但是代码不开源很烦。


## Whether and When does Endoscopy Domain Pretraining Make Sense?
作者：Dominik Batic, Nassir Navab  Computer Aided Medical Procedures, Technical University Munich, Garching, Germany

地址：https://arxiv.org/pdf/2303.17636.pdf

总结：
1. 单独做一个手术领域的预训练模型，不用自然场景下的预训练模型。
（这个想法还是很自然的，就像之前很多做其他类型医学图像的也都有做自己领域的预训练，但是效果都和ImageNet差不多，甚至不如。可能这就涉及到large-scale pretraining到底学习到了什么的理论问题）
2. 收集了一个Endo700k数据集，比ImageNet-1k稍小。之后train from scratch 了一个ViT。方法是MAE。重建的效果跟SimMIM结论有类似，当器械完全被盖住的时候，重建不出来，当有露出的时候，可以重建
![image](https://user-images.githubusercontent.com/109274751/234739627-6b0bb625-2c4f-40e8-a238-074560a5cffa.png)
![image](https://user-images.githubusercontent.com/109274751/234739667-abb7fdb6-210d-478e-b57a-920cf8576282.png)
3. domain-specific pretraining在下游任务更加复杂时效果更好，例如action triplet recognition，而在简单任务（例如phase recognition）上不如ImageNet
4. action triplet recognition setting: 
在triplet任务上，数据集是CholecT45，模型为backbone+linear head。测试集用的是5 videos，应该据是CholecTriplet2021的划分方式。这样做可以用前45个视频用来pretrain。
没有network的细节，例如用没用多任务框架，也没有focal loss和bce的结果对比。效果比ViT/ImageNet好两个点左右
5. phase recognition setting：
数据集是Cholec80，方法是TeCNO，也就是替换spatial backbone，之后用一个MSTCN
使用Full dataset训练时，效果和ImageNet差不多。作者认为是数据集本身够大，能够克服预训练差异
使用few-shot训练时，效果更好但是也差不多相差0.5%-1.5%。可能是重建任务不适合这种时间维度上的phase recognition


## Self-distillation for surgical action recognition
作者：Amine Yamlah，Lena Maier-Hein DFKZ

地址：https://arxiv.org/pdf/2303.12915.pdf

总结：用self-distillation解决class imbalance和label ambiguity。
![image](https://user-images.githubusercontent.com/109274751/234743898-d8b09ee4-7487-4072-a184-76202a5d1178.png)
网络结构：
base model: Swin，最后一层换成节点数100的全连接
+multi: 同时输出instrument,verb,target,phase作为辅助任务
+selfd: teacher model: 使用swin在one-hot label上train 20 epochs，bce loss。训练后使用sigmoid输出训练student model

为什么soft label和自蒸馏会提升模型性能？
soft label可以解决标签错误或者模棱两可的问题。相比于数据集给出的错误标签，soft label可能更接近真正的标签，因此可以使得模型学习到更接近正确的知识


