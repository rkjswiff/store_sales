FROM --platform=linux/amd64 ubuntu

ENV PATH="root/miniconda3/bin:${PATH}"
ARG PATH="root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh \
    && sh Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -b \
    && rm Miniconda3-py38_23.3.1-0-Linux-x86_64.sh

RUN conda create -n mlbasic python=3.6

COPY . src/

RUN /bin/bash -c "source activate mlbasic && pip install pandas scikit-learn"
