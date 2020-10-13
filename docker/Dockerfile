# https://aws.amazon.com/cn/releasenotes/available-deep-learning-containers-images/
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-training:1.6.0-gpu-py36-cu101-ubuntu16.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         libsm6 \
         libxrender1 \
         libglib2.0-dev \
         libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install boto3 opencv-python requests gluoncv==0.8

# copy pretrained model to container
COPY mxnet /root/.mxnet/

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
