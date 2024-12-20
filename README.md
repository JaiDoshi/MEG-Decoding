Preprocessing 

To create the train test splits of the MEG dataset, specify the paths to the preprocessed MEG images and the image embeddings in cells 3, 4 and 7 and run the rest of the cells. The train and test datasets will be created in files 'valid_epochs_small_train_resplit.pickle' and 'valid_epochs_small_train_resplit.pickle', and the corresponding embeddings will be created in files 'image_embeddings_vit_small_train_resplit.npy' and 'image_embeddings_vit_test_small_resplit.npy' as dictionaries. Similarly, use the files create_meg_fmri_subset.ipynb and create_meg_fmri_combined.ipynb to create the subset of MEG images and the combination of MEG and fMRI images for the multimodal experiments. 



Clustering 

The paths to the preprocessed MEG embeddings from the original dataset have to be set in cell 2 and the path to the tsv file containing the high-level THINGS categories has to be set in cell 11. The rest of the cells can be run as is. 
