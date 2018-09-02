# JDDA
## Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation

This repository contains code for reproducing the experiments reported in the [paper](https://arxiv.org/abs/1808.09347) **Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation**. In the source code, the JDDA_C is denoted by 'center_loss' in Digital_JDDA_C and the  JDDA_C is denoted by 'Manifold' in Digital_JDDA_I.

## train JDDA
This code requires Python 2 and implemented in Tensorflow 1.9. You can download all the datasets used in our paper from [here](https://pan.baidu.com/s/1IMUVnpM8Ve6XX37rtv2zJQ) and place them in the specified directory.
#### Digital Domain Adaptation
```
cd Digital_JDDA_I or Digital_JDDA_C
python trainLenet.py
```
#### Office-31 Domain Adaptation
- Create a txt file for the image path of the Office dataset as shown in [amazon.txt](https://github.com/A-bone1/JDDA/tree/master/Office_JDDA_C/data)
- Download the [ResNet-L50](https://pan.baidu.com/s/1IMUVnpM8Ve6XX37rtv2zJQ) pre-training model
-
```
cd Office_JDDA_C or Office_JDDA_I/JDDA_I
python train_JDDA.py
```

## train Compared Approaches
We mainly compare our proposal with [DDC](https://arxiv.org/abs/1412.3474), [DAN](http://proceedings.mlr.press/v37/long15.pdf),[DANN](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf), [CMD](https://arxiv.org/abs/1702.08811), [ADDA](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf) and [CORAL](https://arxiv.org/abs/1607.01719)

If you want to use these methods, you can modify them in [trainLenet.py](https://github.com/Abone1/JDDA/blob/master/Digital_JDDA_C/trainLenet.py).
```python
93        self.CalDiscriminativeLoss(method="CenterBased")
94        self.CalDomainLoss(method="CORAL")
```

or in [train_JDDA.py](https://github.com/A-bone1/JDDA/blob/master/Office_JDDA_C/JDDA_C/train_JDDA.py)

```python
84    # domain_loss=tf.maximum(0.0001,KMMD(source_model.avg_pool,target_model.avg_pool))
85    domain_loss=coral_loss(source_model.avg_pool,target_model.avg_pool)
86    centers_update_op,discriminative_loss=CenterBased(source_model.avg_pool,y)
87    # domain_loss = mmatch(source_model.avg_pool,target_model.avg_pool, 5)
88  # domain_loss = log_coral_loss(source_model.adapt, target_model.adapt)
```

The results (accuracy %) for unsupervised domain adaptation can be seen here
![image](https://github.com/A-bone1/JDDA/blob/master/img/accuracy.png)

##  t-SNE visualization
If you want to use t-SNE visualization, you can open the comment in [train_JDDA.py](https://github.com/A-bone1/JDDA/blob/master/Digital_JDDA_I/trainLenet.py)
```python
94  # self.conputeTSNE(step, self.SourceData,  self.TargetData,self.SourceLabel, self.TargetLabel, sess)
```


![image](https://github.com/A-bone1/JDDA/blob/master/img/tsne.png)
