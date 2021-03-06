#FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
#FROM floydhub/pytorch:1.0.0-gpu.cuda9cudnn7-py3.39
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

#ARG http_proxy="wwwproxy.se.axis.com:3128"
#ARG https_proxy="wwwproxy.se.axis.com:3128"
#RUN echo 'Acquire::http { Proxy "http://apt-proxy.se.axis.com:8000"; };' >> /etc/apt/apt.conf.d/80axisproxy

RUN apt-get update
RUN apt-get --yes install libglib2.0-0 libsm6 \
                          libxrender1 libavcodec-dev libswscale-dev \
                          libglpk-dev zip libhdf5-dev cmake \
                          && apt-get clean

RUN conda install --yes opencv
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

WORKDIR /workspace
CMD /bin/bash

COPY ggdtrack /workspace/ggdtrack
COPY setup.py /workspace/
COPY test /workspace/test
COPY full_run.py /workspace/
COPY frossard_run.py /workspace/
COPY eval_saved.py /workspace/
COPY eval_saved_hamming.py /workspace/
COPY data_run.py /workspace/
COPY ablation.py /workspace/
RUN python setup.py install


