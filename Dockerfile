FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y wget tar
# RUN wget -O 12_lead_ECGFounder.pth "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth?download=true"

# Create a dedicated folder for downloaded data inside the image
RUN mkdir -p /challenge/downloaded_data

# # Download the challenge datasets into the new folder
# RUN wget -O /challenge/downloaded_data/CODE15_data_part_aa "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/CODE15_data_part_aa?download=true" \
#     && wget -O /challenge/downloaded_data/CODE15_data_part_ab "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/CODE15_data_part_ab?download=true" \
#     && wget -O /challenge/downloaded_data/CODE15_data_part_ac "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/CODE15_data_part_ac?download=true" \
#     && wget -O /challenge/downloaded_data/CODE15_data_part_ad "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/CODE15_data_part_ad?download=true"

# # combine the CODE15 data parts into a single file
# RUN cat /challenge/downloaded_data/CODE15_data_part_* > /challenge/downloaded_data/CODE15_data.hdf5
# RUN rm /challenge/downloaded_data/CODE15_data_part_*

# Download finetune datasets
RUN wget -O /challenge/downloaded_data/PTBXL_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/PTBXL_data.hdf5?download=true"
RUN wget -O /challenge/downloaded_data/SaMiTrop_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/SaMiTrop_data.hdf5?download=true"

# Download external datasets
RUN wget -O /challenge/downloaded_data/CSPC_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/CSPC_data.hdf5?download=true"
RUN wget -O /challenge/downloaded_data/CSPC_extra_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/CSPC_extra_data.hdf5?download=true"
RUN wget -O /challenge/downloaded_data/Chapman_Shaoxing_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/Chapman_Shaoxing_data.hdf5?download=true"
RUN wget -O /challenge/downloaded_data/Georgia_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/Georgia_data.hdf5?download=true"
RUN wget -O /challenge/downloaded_data/Ningbo_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/Ningbo_data.hdf5?download=true"
RUN wget -O /challenge/downloaded_data/PTB_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/PTB_data.hdf5?download=true"
RUN wget -O /challenge/downloaded_data/ST_Petersburg_data.hdf5 "https://huggingface.co/datasets/xiaoyuwang123/CinCo_Amigos_PhysioNet2025/resolve/main/ST_Petersburg_data.hdf5?download=true"


## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
