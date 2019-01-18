FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN apt-get update
RUN apt-get --yes install libglib2.0-0 libopencv-core2.4v5 libsm6 \
                          libxrender1 libavcodec-dev libswscale-dev \
                          libglpk-dev

COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

COPY ggdtrack /workspace/ggdtrack
COPY setup.py /workspace/
COPY test /workspace/test
RUN python setup.py install

RUN apt-get --yes remove libavcodec-dev libswscale-dev libglpk-dev
RUN apt-get clean

