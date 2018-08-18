# alexkim205/keras_music - Keras in Docker with Python 3 and TensorFlow on CPU

FROM alexkim205/keras:0.0.3-cpu
MAINTAINER Alex Kim "alexgkim205@gmail.com"

# install debian packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    fluidsynth \
    pulseaudio \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# install additional python packages
RUN pip3 --no-cache-dir install \
    music21 \
    pyFluidSynth \
    aubio \
    requests \
    spotipy 


RUN echo enable-shm=no >> /etc/pulse/client.conf