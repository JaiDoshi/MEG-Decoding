#!/usr/bin/env python
# coding: utf-8


import argparse
import numpy as np
import mne
import os
from functools import partial 
import pandas as pd



dir_name = '/scratch/jd5697/cv_project'

def main(sub):

    # Assumes you downloaded the single trial responses in table format to this directory 
    betas_csv_dir = os.path.join(dir_name, 'betas_csv')

    data_file = os.path.join(betas_csv_dir, f'sub-{sub}_ResponseData.h5')
    responses_fmri = pd.read_hdf(data_file)  # this may take a minute
    print('Single trial response data')
    responses_fmri.head()

    # Stimulus metadata
    stim_f = os.path.join(betas_csv_dir, f'sub-{sub}_StimulusMetadata.csv')
    stimdata = pd.read_csv(stim_f)
    stimdata.head()


    #Transpose the dataframe and remove the first row
    responses_fmri = responses_fmri.transpose()[1:]
    print(responses_fmri.head())
    print(responses_fmri.shape)


    #Convert the rows into numpy arrays
    responses_fmri['fmri'] = responses_fmri.apply(lambda row: row.to_numpy(), axis=1)

    # Keep only the new column
    responses_fmri = responses_fmri[['fmri']]

    print(responses_fmri.head())
    print(responses_fmri.shape)
    print(responses_fmri['fmri'][0].shape)


    responses_fmri = pd.concat([responses_fmri, stimdata], axis=1)
    print(responses_fmri.head())
    print(responses_fmri.shape)


    # Remove duplicate stimuli from the dataframe
    responses_fmri = responses_fmri.drop_duplicates(subset='stimulus')

    print(responses_fmri.shape)


    fmri_data = np.vstack(responses_fmri['fmri'])
    print(fmri_data.shape)

    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=15000)
    svd.fit(fmri_data)


    print(sum(svd.explained_variance_ratio_))

    # Add the SVD data to the fmri_data dataframe
    fmri_data_svd = svd.transform(fmri_data)
    print(fmri_data_svd.shape)
    responses_fmri['svd'] = [row for row in fmri_data_svd]

    print(type(responses_fmri['svd'][0]))

    print(responses_fmri.head())
    print(responses_fmri.shape)

    # Drop the fmri column
    responses_fmri = responses_fmri.drop(columns=['fmri'])
    
    # Reset the index
    responses_fmri = responses_fmri.reset_index(drop=True)

    # Save the dataframe as a pickle file
    responses_fmri.to_pickle(os.path.join(dir_name, f'sub-{sub}_responses_fmri_svd.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str)
    args = parser.parse_args()
    main(args.sub)