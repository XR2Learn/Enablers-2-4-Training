FROM python:3.10
RUN apt-get update \
&& apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        vim
COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /app/setup.py
RUN mkdir -p /app/handcrafted_features_extraction_audio_modality/ && \
    touch /app/handcrafted_features_extraction_audio_modality/__init__.py
WORKDIR /app
RUN pip install -r requirements.txt && \
    rm -rf /tmp/pip* /root/.cache/pip
ADD ./ /app
RUN pip install -e . && \
    rm -rf /tmp/pip* /root/.cache/pip