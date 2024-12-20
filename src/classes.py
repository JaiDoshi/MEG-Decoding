import torch
from torch.utils.data import Dataset
import numpy as np

class MEGDataset(Dataset):
    def __init__(self, meg_epochs, image_embeddings, layout):
        # self.meg_epochs = meg_epochs
        self.metadata = meg_epochs.metadata # Shape: (B)
        self.meg_data = meg_epochs.get_data()  # Shape: (B, 272, 181)
        self.image_embeddings = image_embeddings  # Shape: (B, 768)
        #self.embedding_indices = self.metadata['embedding_index'].values # (B,)
        self.subject_indices = self.metadata['subject_index'].values # shape (B)
        self.layout = layout
        # Get channel positions
        pos = np.array([self.layout.pos[self.layout.names.index(ch[:5])] 
                                for ch in meg_epochs.info['ch_names']])
        ps = pos[:,:2]
        x = (ps[:, 0] - ps[:, 0].min()) / (ps[:, 0].max() - ps[:, 0].min())
        y = (ps[:, 1] - ps[:, 1].min()) / (ps[:, 1].max() - ps[:, 1].min())
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        ps[:,0] = x
        ps[:,1] = y
        self.ch_pos = ps.copy()

        
    def __len__(self):
        return len(self.meg_data) # 19848
    
    def __getitem__(self, idx):
        # meg_epochs = meg_epochs[idx]
        meg = torch.FloatTensor(self.meg_data[idx])
        metadata = self.metadata.iloc[idx, :].to_dict()
        image_embeddings = torch.FloatTensor(self.image_embeddings[idx])
        subject_index = self.subject_indices[idx]
        ch_pos = torch.FloatTensor(self.ch_pos)
        

        return {
            # 'meg_epochs': meg_epochs,
            'meg': meg,
            'metadata': metadata,
            'image_embeddings': image_embeddings,
            'subject_index': subject_index,
            'ch_pos': ch_pos
        }
        
        