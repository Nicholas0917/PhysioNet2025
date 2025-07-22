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


## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
