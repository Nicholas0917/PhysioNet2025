FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y wget tar
RUN wget -O 12_lead_ECGFounder.pth "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth?download=true"
RUN mkdir -p ./tmp \
    && wget -O ./tmp/ECG_signal.tar.gz "https://leeds365-my.sharepoint.com/personal/wmqn2362_leeds_ac_uk/_layouts/52/download.aspx?share=EVgqmhTaJZBAjchf58WIv7EBhvLE4Q1BvHkslZkn994SSA" \
    && tar -xzf ./tmp/ECG_signal.tar.gz -C ./tmp \
    && rm ./tmp/ECG_signal.tar.gz

RUN wget -O ./tmp/CSPC_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/CSPC_data.hdf5?download=true"
RUN wget -O ./tmp/CSPC_extra.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/CSPC_extra_data.hdf5?download=true"
RUN wget -O ./tmp/Chapman_Shaoxing_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/Chapman_Shaoxing_data.hdf5?download=true"
RUN wget -O ./tmp/Georgia_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/Georgia_data.hdf5?download=true"
RUN wget -O ./tmp/Ningbo_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/Ningbo_data.hdf5?download=true"
RUN wget -O ./tmp/PTB_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/PTB_data.hdf5?download=true"
RUN wget -O ./tmp/ST_Petersburg_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/ST_Petersburg_data.hdf5?download=true"


## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
