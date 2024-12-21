## Project Structure

### 1. **Preprocessing**

#### 1.1 MEG Preprocessing
- **[create_meg_dataset.ipynb](src/preprocessing/create_meg_dataset.ipynb)**  
Run the file `MEG_preprocessing.ipynb` after pointing the variables specifying the path to the MEG dataset. 

#### 1.2 fMRI Preprocessing 
- **[fmri_dimensionality_reduction.ipynb](src/preprocessing/fmri_dimensionality_reduction.ipynb)**  
To create the dimensionality-reduced fMRI data, run the script `fmri_dimensionality_reduction.py`. Specify the dir_name and the path the fMRI data in the script, and pass the sub ('01', '02', '03') as an argument. 

#### 1.3 MEG and Multimodal Preprocessing 
- **[create_meg_fmri_subset.ipynb](src/preprocessing/create_meg_fmri_subset.ipynb)**
- **[create_meg_fmri_dataset.ipynb](src/preprocessing/create_meg_fmri_dataset.ipynb)**  
To create the train test splits of the MEG dataset, specify the paths to the preprocessed MEG images and the image embeddings in cells 3, 4 and 7 and run the rest of the cells. The train and test datasets will be created in files `valid_epochs_small_train_resplit.pickle` and `valid_epochs_small_test_resplit.pickle`, and the corresponding RGB embeddings will be created in files `image_embeddings_vit_small_train_resplit.npy` and `image_embeddings_vit_test_small_resplit.npy` as dictionaries. Similarly, use the files `src/preprocessing/create_meg_fmri_subset.ipynb` and `src/preprocessing/create_meg_fmri_combined.ipynb` to create the subset of MEG images and the combination of MEG and fMRI images for the multimodal experiments. 

### 2. **Regression**
- **[regression_fmri_all_participants.ipynb](regression/regression_fmri_all_participants.ipynb)**  
- **[regression_meg_all_participants.ipynb](regression/regression_meg_all_participants.ipynb)**  
- **[regression_meg_all_participants_fmri_subset.ipynb](regression/regression_meg_all_participants_fmri_subset.ipynb)**  
- **[regression_meg_fmri_all_participants_combined.ipynb](regression/regression_meg_fmri_all_participants_combined.ipynb)**  

Set the appropriate paths from preprocessing in the starting cells, then run the file `regression_meg.ipynb` for linear regression on the MEG dataset, and the files `regression_meg_fmri_subset.ipynb`, `regression_fmri.ipynb`, and `regression_meg_fmri_combined.ipynb` for linear regression on the MEG subset, fMRI images and combined data respectively.

### 3. **Clustering**

The paths to the preprocessed MEG embeddings from the original dataset have to be set in cell 2 and the path to the tsv file containing the high-level THINGS categories has to be set in cell 11. The rest of the cells can be run as is. 

### 3. **Neural Models**


#### 3.1 Brain Decoding

#### 3.2 Vision Transformer



