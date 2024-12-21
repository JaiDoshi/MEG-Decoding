import os
import argparse
from tqdm import tqdm
import pickle
import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from simpleconv_reg import SimpleConv
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
from classes import MEGDataset
from utils import soft_clip_loss, hard_clip_loss, calculate_params, train_modified, val_modified, test_model
import time
import functools
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MEG model with configurable parameters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--early_stopping", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--lr_schedule", type=str, default="linear", help="Learning rate schedule.", choices=["linear", "cyclic"]) # Cyclic to be implemented
    parser.add_argument("--warmup_lr", type=float, default=None, help="Warm-up learning rate.")
    parser.add_argument("--warmup_interval", type=int, default=None, help="Warm-up interval in iterations.")
    parser.add_argument("--loss_func", type=str, default="hard_clip_loss", help="Loss function to use.", choices=["soft_clip_loss", "hard_clip_loss"])
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs.")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every n epochs.")
    parser.add_argument("--print_interval", type=int, default=50, help="Print training loss every n steps.")
    parser.add_argument("--dilation_type", type=str, default="expo", help="Type of dilation growth. expo/cyclic", choices=["expo", "cyclic"])
    parser.add_argument("--dropout", type=float, default=0.2, help=" Dropout for ProjectionHead and ConvBlock.")
    parser.add_argument("--embeddings_type", type=str, default="vit", help="Type of embeddings to use. vit/dino", choices=["vit", "dino"])
    parser.add_argument("--dataset_type", type=str, default="large", help="Type of dataset to use. large/small", choices=["large", "small"])
    parser.add_argument("--preprocessing_type", type=str, default="raj", help="Type of preprocessing to use. raj/direct", choices=["raj", "direct"])
    return parser.parse_args()


# need to change paths
if __name__ == "__main__":
    args = parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using DEVICE:", DEVICE)
    
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.wandb_run_name if args.wandb_run_name else f"BM_PreRaj_Drpt{args.dropout}_Diln{args.dilation_type}_ClipEmb{args.embeddings_type}_Loss{args.loss_func}_B{args.batch_size}_LR{args.lr}_S{args.seed}_E{args.epochs}_{now}"
    args.wandb_run_name = run_name
    
    
    # Load Valid Epochs 
    if args.dataset_type == "small" and args.preprocessing_type == "raj": 
        with open('valid_epochs_raj_adjusted_train_bd.pickle', 'rb') as f:
            valid_epochs = pickle.load(f)
        with open('valid_epochs_raj_small_test_bd.pickle', 'rb') as f:
            valid_epochs_test = pickle.load(f)
    
    elif args.dataset_type == "large" and args.preprocessing_type == "raj":
        with open('valid_epochs_raj_adjusted_train_bd.pickle', 'rb') as f:
            valid_epochs = pickle.load(f)
        with open('valid_epochs_raj_large_test_bd.pickle', 'rb') as f:
            valid_epochs_test = pickle.load(f)
            
    elif args.preprocessing_type == "direct":
        raise NotImplementedError("Direct preprocessing not implemented yet.")
        # with open('valid_epochs_direct_all_train.pickle', 'rb') as f:
        #     valid_epochs = pickle.load(f)
        # with open('valid_epochs_direct_all_test.pickle', 'rb') as f:
        #     valid_epochs_test = pickle.load(f)
    # Load Embeddings
    if args.embeddings_type == "vit": 
        raise NotImplementedError("Vit embeddings not implemented yet.")
        # if args.dataset_type == "small":
        #     embeddings = np.load('./image_embeddings_vit_small_train_resplit.npy', allow_pickle=True).item()
        #     embeddings_test = np.load('./image_embeddings_vit_small_test_resplit.npy', allow_pickle=True).item()
        #     embeddings_val = np.array([embeddings[filename] for filename in valid_epochs.metadata['image_path']])
        #     embeddings_val_test = np.array([embeddings_test[filename] for filename in valid_epochs_test.metadata['image_path']])
        
        # elif args.dataset_type == "large": # Coz large dataset has filenames with extra '/' in the beginning
        #     embeddings = np.load('./image_embeddings_vit_train.npy', allow_pickle=True).item()
        #     embeddings_test = np.load('./image_embeddings_vit_test.npy', allow_pickle=True).item()
        #     embeddings_val = np.array([embeddings[filename[1:]] for filename in valid_epochs.metadata['image_path']])
        #     embeddings_val_test = np.array([embeddings_test[filename[1:]] for filename in valid_epochs_test.metadata['image_path']])
        
    elif args.embeddings_type == "dino":
        embeddings = np.load('./dinov2_embeddings_adjusted_train_dict.npy', allow_pickle=True).item()
        
        if args.dataset_type == "small":
            embeddings_test = np.load('./dinov2_embeddings_small_test_dict.npy', allow_pickle=True).item()
            embeddings_test = {os.path.basename(k): v for k, v in embeddings_test.items()}
            embeddings_val = np.array([embeddings[filename] for filename in valid_epochs.metadata['image_path']])
            embeddings_val_test = np.array([embeddings_test[os.path.basename(filename)] for filename in valid_epochs_test.metadata['image_path']])
        elif args.dataset_type == "large":
            embeddings_test = np.load('./dinov2_embeddings_large_test_dict.npy', allow_pickle=True).item()
            embeddings_val = np.array([embeddings[filename] for filename in valid_epochs.metadata['image_path']])
            embeddings_val_test = np.array([embeddings_test[filename] for filename in valid_epochs_test.metadata['image_path']])
        


    #VERIFY THINGS:
    # Convert both to sets
    filenames_set1 = set(valid_epochs.metadata['image_path'])
    filenames_set2 = set(embeddings.keys())

    # Check if the sets are equal
    if filenames_set1 == filenames_set2:
        print("All filenames match!")
    else:
        print("Filenames do not exactly match.")
        # print("Missing in embeddings:", filenames_set1 - filenames_set2)
        # print("Extra in embeddings:", filenames_set2 - filenames_set1)
        
    layout = mne.channels.find_layout(valid_epochs.info, ch_type="meg")
    layout_test = mne.channels.find_layout(valid_epochs_test.info, ch_type = 'meg')
    
    # embeddings_val = np.array([embeddings[filename] for filename in embeddings.keys()])
    # embeddings_val_test = np.array([embeddings_test[filename] for filename in embeddings_test.keys()])
    
    # TEST THINGS
    print("Train Valid Epochs Shape:",valid_epochs.get_data().shape)
    print("Test Valid Epochs Shape:",valid_epochs_test.get_data().shape)
    
    print("Working on Data Loaders..")

    #TRAIN DATASET`
    dataset = MEGDataset(valid_epochs, embeddings_val, layout)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=1)

    # TEST DATASET
    test_dataset = MEGDataset(valid_epochs_test, embeddings_val_test, layout_test)
    if args.dataset_type == "small":
        test_batch_size = 200
    elif args.dataset_type == "large":
        test_batch_size = 2400
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, num_workers=1)
    print("Data Loaders Ready, Instantiating Model.. ")
    
    if args.preprocessing_type == "direct":   
        in_channels = 271
        temporal_dim = 281
    elif args.preprocessing_type == "raj":
        in_channels = 272 
        temporal_dim = 181
    
    model = SimpleConv(
        in_channels=in_channels,
        out_channels=2048,
        merger_dropout=0.2,
        hidden_channels=320,
        n_subjects=4,
        merger=True,
        merger_pos_dim=2048,
        subject_layers=True,
        subject_layers_dim="input",
        gelu=True,
        device=DEVICE,
        dilation_type=args.dilation_type,
        extra_dropout=args.dropout,
        temporal_dim=temporal_dim
    ).to(DEVICE)
    print("Model Instantiated!, Model device:", model.device)
    # print("Model device:", model.device)

    trained_model, train_loss, val_loss, best_model_path = train_modified(
        model, train_loader, val_loader, args, DEVICE
    )

    # if args.wandb_project:
    #     # now = time.strftime("%Y-%m-%d_%H-%M-%S")
    #     now = "2024-12-11_14-57-04"
    #     run_name = args.wandb_run_name if args.wandb_run_name else f"BM_B_{args.batch_size}_LR_{args.lr}_S_{args.seed}_E_{args.epochs}__{now}"
    #     wandb.init(project=args.wandb_project, name=run_name)
    #     wandb.config.update(vars(args))

    print("*********** Model Trained! Best Model Path", best_model_path)
    # best_model_path = "./output/BM_PreRaj_Drpt0.3_Dilnexpo_ClipEmbvit_Losssoft_clip_loss_B128_LR0.0003_S42_E100_2024-12-09_12-43-25/best_model.pth"

    test_loss, test_top1_accuracy, test_top5_accuracy = test_model(model, test_loader, best_model_path, args, DEVICE)


