# **Two-Stage Domain Adversarial Learning to Identify Chagas Disease from ECG and Patient Demographic Data**

## **Introduction**

This repository contains the code for the submission by our team, CinCo Amigos, to the [George B. Moody PhysioNet Challenge 2025](https://physionetchallenges.org/2025/). Our goal was to develop an automated, open-source algorithm to detect Chagas disease using electrocardiograms (ECGs) and patient demographic data.

Chagas disease is widely underdiagnosed due to limited serological test coverage. Large-scale automated ECG screening offers a promising solution. However, this task presents significant challenges, including:

* **Significant Label Noise**: The largest dataset (CODE-15) contains unreliable self-reported labels, whereas smaller datasets provide reliable annotations.  
* **Extreme Class Imbalance**: The prevalence of the positive class is only 2%.  
* **Substantial Domain Shift**: There is a noticeable performance drop between internal testing and public scoring metrics, indicating differences between data sources.

To address these challenges, we proposed a **Two-Stage Domain Adversarial Learning** approach. This framework combines a custom neural network architecture with noise-robust learning techniques, domain-adversarial methods, and advanced class-imbalance handling strategies.

## **Methodology Overview**

Our approach follows a two-stage training paradigm, illustrated in Fig. 2 of the paper:
<p align="center">  <img width="705" height="417" alt="image" src="https://github.com/user-attachments/assets/796f5c9a-68d1-4a62-9739-420b82d54af9" />



1. **Stage 1: Pre-training with Noise and Domain Adaptation**  
   * A custom neural network (an encoder based on ECGNeXt or SEResNet, plus a Meta Net for demographic covariates) is pre-trained on the large, noisy CODE15 dataset.  
   * **LMFLoss**, a combination of Focal Loss and Label-Distribution-Aware Margin (LDAM) Loss, is used to handle class imbalance.  
   * **Early Learning Regularization (ELR)** is integrated to counteract label noise.  
   * **Domain-Adversarial Neural Network (DANN)** is employed, incorporating several external datasets (e.g., CSPC, PTB, etc., ignoring their diagnostic labels) as distinct domains to learn domain-invariant features. The encoder is trained to confuse a domain classifier, forcing it to learn domain-agnostic representations.  
2. **Stage 2: Fine-tuning with Preservation of Domain Generalisation**  
   * The model is adapted using smaller, high-quality datasets (e.g., SaMi-Trop, PTB-XL).  
   * **Feature Distillation** is used, where the pre-trained encoder acts as a frozen "teacher" model guiding the "student" encoder during fine-tuning to retain domain generalisation capabilities.  
   * Alternatively, the pre-trained encoder is frozen, and only the classifier head is fine-tuned.


## **How to Run the Code**

### **1\. Environment Setup**

You can set up the environment in one of two ways:

* Using Docker (Recommended):  
  Build the Docker image:
  ```
  docker build -t cinc2025-image .
  ```
  This command uses the Dockerfile to build an image containing all dependencies. The Dockerfile also automatically downloads the required external datasets into the /challenge/downloaded\_data directory within the image.  
* Using a Python Virtual Environment:  
  Create and activate a virtual environment (e.g., using venv or conda), then install the required dependencies:
  ```
  pip install -r requirements.txt
  ```
  You will need to ensure the necessary datasets are downloaded and accessible. The Dockerfile indicates the HDF5 files required.

### **2\. Data Preparation (If not using pre-downloaded data in Docker)**

* **Download Datasets**: Obtain the required ECG datasets (CODE-15%, SaMi-Trop, PTB-XL, CSPC, PTB, etc.) from the [PhysioNet Challenge 2025 website](https://physionetchallenges.org/2025/#data) and other sources cited in the paper.  
* **Preprocessing**: Run the relevant data preparation scripts to convert the raw data into HDF5 or WFDB format. Scripts provided include prepare\_code15\_data.py, prepare\_samitrop\_data.py, prepare\_ptbxl\_data.py, and Prepare\_External\_data.py.  
  * Example for CODE-15% data to WFDB:  
    ```
    python prepare_code15_data.py -i <path_to_input_hdf5_files> -d <path_to_demographics_csv> -l <path_to_labels_csv> -o <output_wfdb_folder>
    ```
  * Example for preparing external datasets into HDF5:  
    ```
    python Prepare_External_data.py --data_dir <raw_data_directory> --output_path <output_hdf5_file_path>
    ```

### **3\. Training the Model**

Use the train\_model.py script to train the model. If running inside a Docker container, make sure to mount your local data and model folders.
```
python train_model.py -d <path_to_training_data_folder> -m <path_to_model_output_folder> -v
```
* `<path_to_training_data_folder>`: Directory containing the training data files (likely preprocessed .hdf5 files based on Dockerfile and dataset.py).  
* `<path_to_model_output_folder\>`: Directory where the trained model(s) will be saved.  
* `-v`: (Optional) Enable verbose output.

This script executes the train\_model function in team\_code.py, which implements the two-stage training strategy described in the paper.

### **4\. Running the Model for Prediction**

Use the run\_model.py script to make predictions on new data.
```
python run_model.py -d <path_to_test_data_folder> -m <path_to_model_folder> -o <path_to_output_folder> -v
```
* `<path_to_test_data_folder>`: Directory containing the data files for prediction (expects WFDB format .hea/.dat or .mat files).  
* `<path_to_model_folder>`: Directory containing the trained model(s) saved by train\_model.py.  
* `<path_to_output_folder>`: Directory where the model's predictions (one .txt file per record) will be saved.  
* `-v`: (Optional) Enable verbose output.

This script calls the load\_model and run\_model functions from team\_code.py.

### **5\. Evaluating Model Performance**

Use the official evaluate\_model.py script (or the version included in this repository) to evaluate the model's performance.
```
python evaluate_model.py -d <path_to_labeled_data_folder> -o <path_to_model_output_folder> -s <path_to_scores_file>
```
* `<path_to_labeled_data_folder>`: Directory containing the ground truth label files for the test data.  
* `<path_to_model_output_folder>`: Directory containing the model's predictions generated by run\_model.py.  
* `<path_to_scores_file>`: (Optional) Path to save the evaluation scores in a CSV file.

The script will compute and output the challenge metrics, such as the Challenge score, AUROC, AUPRC, etc..

## **Results**

Our approach achieved a mean Challenge score of 0.250 on the official hidden test sets of the PhysioNet Challenge 2025, ranking 7th out of 40 competing teams. Notably, our model ranked 1st on the ELSA-Brasil test set. 
<p align="center"> <img width="576" height="250" alt="image" src="https://github.com/user-attachments/assets/3de62d9d-f0ce-4a3d-ba13-33b211df9809" />


## **Citation**

If you use this code or methodology in your research, please cite our paper:

Wang, X., Syversen, A., Ding, Z., Battye, J., Ho, S. Y. S., & Wong, D. C. (2025). Two-Stage Domain Adversarial Learning to Identify Chagas Disease from ECG and Patient Demographic Data. *Computing in Cardiology* 

And the relevant PhysioNet Challenge 2025 papers.

## **Contact**

For any questions, please contact Xiaoyu Wang (wmqn2362@leeds.ac.uk).
