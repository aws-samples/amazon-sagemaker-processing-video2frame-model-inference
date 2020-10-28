# Amazon SageMaker Processing Video2frame Model Inference

This demo shows how to use SageMaker processing process video frames extraction and model inference.

Some business scenario need to processing videos by using machine learning. They usually need extract frames from videos and then send them to models and get the result. This need you extract the frames and store in some place and then using batch transformer or online inference, which would involve a storage cost which is no longer need after inference. So customers are looking for a way to finish such job in a effective way, here we would introduce Amazon SageMaker Processing.

Amazon SageMaker Processing, a new capability of Amazon SageMaker that lets customers easily run the preprocessing, postprocessing and model evaluation workloads on fully managed infrastructure, was announced during re:Invent 2019. 

In this sample, we would lauch a sagemaker processing job in a VPC, the input is videos in S3, and output is inference results (segmentation images) and will be stored in S3.

1. Launch an EC2 instance to play as API server which could be called by sagemaker processing job.
2. We use pretrained model to do semantic segmentation inference from GluonCV model zoo.
3. Enable Sagemaker Processing vpc mode so it could call API server.

Here is the high level architecture of this sample.

![High level architecture](https://sagemaker-demo-dataset.s3-us-west-2.amazonaws.com/Picture1.png)


### GluonCV
---
[GluonCV](https://gluon-cv.mxnet.io/) provides implementations of state-of-the-art (SOTA) deep learning algorithms in computer vision. It aims to help engineers, researchers, and students quickly prototype products, validate new ideas and learn computer vision.

[GluonCV model zoo](https://gluon-cv.mxnet.io/model_zoo/index.html) contains six kinds of pretrained model: Classification, Object Detection, Segmentation, Pose Estimation, Action Recognition and Depth Prediction.

In this sample, we will use **deeplab_resnet101_citys** from Segmentation and was trained with **[cityscape dataset](https://www.cityscapes-dataset.com/)**, which focuses on semantic understanding of urban street scenes, so this model is suitable for car view images.

### Prerequisite
---
In order to download GPU supported pretrianed model, you need run this sample in **GPU based instance**, such as **ml.p2.xlarge or ml.p3.2xlarge**.

If you only launch none gpu instances in processing jobs, such as c5 type, you could run this demo in none gpu based instances.


## Security
 
 See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
 
## License
 
 This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.   
