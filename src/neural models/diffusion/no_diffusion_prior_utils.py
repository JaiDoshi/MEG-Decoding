import torch
import os
import wandb
import time
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
# from transformers import AdamW, get_scheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
import PIL
from PIL import Image

from torchvision import transforms

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def mse_loss(preds, targs):
    preds_scaled = preds*100.0
    targs_scaled = targs*100.0
    return torch.nn.MSELoss()(preds_scaled, targs_scaled)

def hard_clip_loss(preds, targs, temp= 1.0/np.exp(1.0)):
    clip_clip = torch.eye(preds.shape[0]).to(preds.device)
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip).sum(-1).mean()
    # loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    # loss = (loss1 + loss2)/2
    loss = loss1
    return loss



def calculate_params(model):
    # Calculate total and trainable parameters by module
    module_params = {}  # Dictionary to store total parameters by module
    module_trainable_params = {}  # Dictionary to store trainable parameters by module

    for name, param in model.named_parameters():
        # Get the top-level module name only (in case of nested modules)
        module_name = name.split('.', 2)[0]
        
        # Initialize counts if this is the first time the module is encountered
        if module_name not in module_params:
            module_params[module_name] = 0
            module_trainable_params[module_name] = 0
        
        # Count parameters
        module_params[module_name] += param.numel()
        if param.requires_grad:
            module_trainable_params[module_name] += param.numel()
            
    # Print total parameters and trainable parameters for each module
    print("Parameters by Module:")
    for module_name in module_params:
        print(f"{module_name}:")
        print(f"  Total Params: {module_params[module_name]}")
        print(f"  Trainable Params: {module_trainable_params[module_name]}")

    # Overall totals
    total_params = sum(module_params.values())
    trainable_params = sum(module_trainable_params.values())
    print("\nTotal Params in Model: ", total_params)
    print("Total Trainable Params in Model: ", trainable_params)
    
    # Print shapes
    print("\nModel Shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    return total_params, trainable_params

def val_modified(model, dataloader, loss_func, DEVICE):
    model.eval()
    epoch_loss, num_correct_top1, num_correct_top5, total = [], 0, 0, 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            # print(f"Batch {i + 1} of {len(dataloader)}")
            meg = batch['meg'].to(DEVICE)
            embeddings = batch['image_embeddings'].to(DEVICE)
            embeddings_normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
            # print("Norm of Image Emb:", embeddings_normalized.norm()) 
            labels = torch.arange(meg.size(0), dtype=torch.long, device=DEVICE)

            clip_output = model(batch).to(DEVICE)
            meg_normalized = clip_output / clip_output.norm(dim=1, keepdim=True)
            # print("Norm of MEG:", meg_normalized.norm())    
            
            cos_similarities = torch.matmul(meg_normalized, embeddings_normalized.transpose(0, 1))
            if loss_func in (soft_clip_loss, hard_clip_loss, mse_loss):
                loss = loss_func(meg_normalized, embeddings_normalized)
                
            else: loss = loss_func(cos_similarities, labels)
            epoch_loss.append(loss.item())

            _, predicted_top1 = torch.max(cos_similarities, dim=1)
            num_correct_top1 += (predicted_top1 == labels).sum().item()

            _, predicted_top5 = cos_similarities.topk(5, dim=1)
            correct_top5 = predicted_top5.eq(labels.view(-1, 1).expand_as(predicted_top5))
            num_correct_top5 += correct_top5.sum().item()

            total += labels.size(0)

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    top1_accuracy = 100 * num_correct_top1 / total
    top5_accuracy = 100 * num_correct_top5 / total
    return avg_epoch_loss, top1_accuracy, top5_accuracy


def train_modified(model, train_dataloader, val_dataloader, args, DEVICE):
    if args.wandb_project:
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = args.wandb_run_name if args.wandb_run_name else f"BM_B_{args.batch_size}_LR_{args.lr}_S_{args.seed}_E_{args.epochs}__{now}"
        wandb.init(project=args.wandb_project, name=run_name)
        wandb.config.update(vars(args))
    calculate_params(model)
    
    total_train_steps = len(train_dataloader)*args.epochs
    print(f"Total train steps: {total_train_steps}")
    
    warmup_steps = args.warmup_interval
    warmup_lr = args.warmup_lr
    start_factor_w = args.warmup_lr/ args.lr
    num_train_batches = len(train_dataloader)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    print("Warm up learning rate:", warmup_lr)
    print("Warm up steps:", warmup_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor_w, end_factor=1.0, total_iters=warmup_steps)
    linear_decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_train_steps - warmup_steps)
    
    # Combine warmup and linear decay using SequentialLR
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, linear_decay_scheduler], milestones=[warmup_steps])
    
    if args.loss_func == "soft_clip_loss":
        loss_func = soft_clip_loss
    elif args.loss_func == "hard_clip_loss":
        loss_func = hard_clip_loss
    elif args.loss_func == "mse_loss":
        loss_func = mse_loss
    print("Loss Function in Train:", loss_func)
    print("Training started...")
    
    progress_bar = tqdm(range(args.epochs * len(train_dataloader)))

    iteration = 0
    collate_loss, collate_loss_val = [], []
    best_top5_accuracy, patience_counter = 0.0, 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_model_path = os.path.join(output_dir, "best_model.pth")

    for epoch_num in range(args.epochs):
        epoch_loss, train_num_correct_top1, train_num_correct_top5, train_total = [], 0, 0, 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            meg = batch['meg'].to(DEVICE)
            embeddings = batch['image_embeddings'].to(DEVICE)
            embeddings_normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
            
            clip_output = model(batch).to(DEVICE)
            meg_normalized = clip_output / clip_output.norm(dim=1, keepdim=True)

            cos_similarities = torch.matmul(meg_normalized, embeddings_normalized.transpose(0, 1))
            labels = torch.arange(meg.size(0), dtype=torch.long, device=DEVICE)

            if loss_func in (soft_clip_loss, hard_clip_loss, mse_loss):
                loss = loss_func(meg_normalized, embeddings_normalized)
            else:
                loss = loss_func(cos_similarities, labels)
            epoch_loss.append(loss.item())
            
            _, predicted_top1 = torch.max(cos_similarities, dim=1)
            train_num_correct_top1 += (predicted_top1 == labels).sum().item()

            _, predicted_top5 = cos_similarities.topk(5, dim=1)
            correct_top5 = predicted_top5.eq(labels.view(-1, 1).expand_as(predicted_top5))
            train_num_correct_top5 += correct_top5.sum().item()

            train_total += labels.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()
            iteration += 1
            progress_bar.update(1)

            if step % args.print_interval == 0:
                print(f"Epoch {epoch_num + 1}, Step {step + 1}: Train Loss = {loss.item():.4f}, LR = {scheduler.get_last_lr()[0]:.7f}")


        avg_tr_loss = sum(epoch_loss) / len(epoch_loss)
        collate_loss.append(avg_tr_loss)
        top1_accuracy_train = 100 * train_num_correct_top1 / train_total
        top5_accuracy_train = 100 * train_num_correct_top5 / train_total
        
        avg_val_loss, top1_acc, top5_acc = val_modified(model, val_dataloader, loss_func, DEVICE)
        collate_loss_val.append(avg_val_loss)
        print(
            f"[Epoch {epoch_num + 1}] Train Loss: {avg_tr_loss:.4f}, TRAIN Top-1 Acc: {top1_accuracy_train:.2f}%, TRAIN Top-5 Acc: {top5_accuracy_train:.2f}%\n"
            f"Val Loss: {avg_val_loss:.4f}, Top-1 Acc: {top1_acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%"
        )

        if args.wandb_project:
            wandb.log({
                "epoch": epoch_num + 1,
                "step": iteration,
                "learning_rate": scheduler.get_last_lr()[0],
                "train_loss": avg_tr_loss,
                "train_top1_accuracy": top1_accuracy_train,
                "train_top5_accuracy": top5_accuracy_train,
                "val_loss": avg_val_loss,
                "val_top1_accuracy": top1_acc,
                "val_top5_accuracy": top5_acc
            })
        print("****** Prev Best Top 5 Val Acc:", best_top5_accuracy)
        print("****** Current Top 5 Val Acc:", top5_acc)
        # Save best model based on Top-5 accuracy
        if top5_acc > best_top5_accuracy:
            best_top5_accuracy, patience_counter = top5_acc, 0
            print("Validation Top-5 accuracy improved !!!. Saving model...")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"*** No improvement in Top-5 accuracy for {patience_counter} epoch(s). ***")

        if args.early_stopping and patience_counter >= args.early_stopping:
            print(f"!!!! Early stopping triggered after {args.early_stopping} epochs of no improvement !!!!")
            break

        if (epoch_num + 1) % args.save_interval == 0:
            model_path = os.path.join(output_dir, f"model_epoch_{epoch_num + 1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    print(f"Training complete. Best model saved at {best_model_path}.")

    progress_bar.close()
    return model, collate_loss, collate_loss_val, best_model_path



def test_model(model, test_dataloader, best_model_path, args, DEVICE):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(args)
    if args.loss_func == "soft_clip_loss":
        loss_func = soft_clip_loss
    elif args.loss_func == "hard_clip_loss":
        loss_func = hard_clip_loss
    elif args.loss_func == "mse_loss":
        loss_func = mse_loss
    print("Loss Function in Test:", loss_func)
    avg_test_loss, top1_accuracy, top5_accuracy = val_modified(model, test_dataloader, loss_func, DEVICE)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Top-1 Accuracy: {top1_accuracy:.2f}%, Test Top-5 Accuracy: {top5_accuracy:.2f}%")

    if args.wandb_project:
        wandb.log({
            "test_loss": avg_test_loss,
            "test_top1_accuracy": top1_accuracy,
            "test_top5_accuracy": top5_accuracy,
        })
    return avg_test_loss, top1_accuracy, top5_accuracy

    
###############################################################
#### FOR  257 * 768 SimpleCONV Model, WITH RECONSTRUCTION ####
###############################################################
def val_new(model, dataloader, loss_func, args, DEVICE, epoch_num, vd_pipe):
    model.eval()
    epoch_loss, num_correct_top1, num_correct_top5, total = [], 0, 0, 0
    epoch_loss_nce = []
    epoch_loss_mse = []
    val_batch0 = None
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            # print(f"Batch {i + 1} of {len(dataloader)}")
            if val_batch0 is None:
                val_batch0 = batch.copy()
            meg = batch['meg'].to(DEVICE)
            embeddings = batch['image_embeddings'].to(DEVICE)
            # embeddings_normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
            embeddings_normalized = nn.functional.normalize(embeddings.flatten(1), dim=-1)
            
            #FORWARD PASS
            clip_head_out, mse_head_out = model(batch)
            clip_head_out = clip_head_out.to(DEVICE)
            mse_head_out = mse_head_out.to(DEVICE)
            # meg_normalized = clip_output / clip_output.norm(dim=1, keepdim=True)
            meg_normalized = nn.functional.normalize(clip_head_out.flatten(1), dim=-1)
            mse_out_normalized = nn.functional.normalize(mse_head_out.flatten(1), dim=-1)

            cos_similarities = torch.matmul(meg_normalized, embeddings_normalized.transpose(0, 1))
            labels = torch.arange(meg.size(0), dtype=torch.long, device=DEVICE)

            if loss_func in (soft_clip_loss, hard_clip_loss, mse_loss):
                loss_nce = loss_func(meg_normalized, embeddings_normalized)
            else:
                loss_nce = loss_func(cos_similarities, labels)
                
            loss_mse = mse_loss(mse_head_out, embeddings)
            
            if args.use_mse and args.use_clip:
                epoch_loss_nce.append(loss_nce.item())
                epoch_loss_mse.append(loss_mse.item())
                loss = (args.nce_mult * loss_nce) + (args.mse_mult * loss_mse)
            elif args.use_clip:
                epoch_loss_nce.append(loss_nce.item())
                loss = loss_nce
                loss_mse = torch.tensor(0.0)
            elif args.use_mse:
                epoch_loss_mse.append(loss_mse.item())
                loss = loss_mse # args.prior_mult * loss_prior
                loss_nce = torch.tensor(0.0)
                
            epoch_loss.append(loss.item())

            _, predicted_top1 = torch.max(cos_similarities, dim=1)
            num_correct_top1 += (predicted_top1 == labels).sum().item()

            _, predicted_top5 = cos_similarities.topk(5, dim=1)
            correct_top5 = predicted_top5.eq(labels.view(-1, 1).expand_as(predicted_top5))
            num_correct_top5 += correct_top5.sum().item()

            total += labels.size(0)

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    if args.use_clip:
        avg_epoch_loss_nce = sum(epoch_loss_nce) / len(epoch_loss_nce)
    else: avg_epoch_loss_nce = 0.0
    if args.use_mse:
        avg_epoch_loss_mse = sum(epoch_loss_mse) / len(epoch_loss_mse)
    else: avg_epoch_loss_mse = 0.0

    top1_accuracy = 100 * num_correct_top1 / total
    top5_accuracy = 100 * num_correct_top5 / total
    
    if epoch_num%1 == 0:
        print('reconstructing...')
        with torch.no_grad():
            vd_pipe = vd_pipe.to(DEVICE)
            grid, _, _, _ = reconstruction_new(
                val_batch0,
                clip_extractor = None,
                unet = vd_pipe.image_unet, 
                vae = vd_pipe.vae, 
                noise_scheduler = vd_pipe.scheduler,
                # diffusion_priors = None,
                num_inference_steps = args.num_inference_steps,
                n_samples_save = 1,
                guidance_scale = args.guidance_scale,
                timesteps_prior = args.timesteps,
                seed = args.seed,
                retrieve = False,
                plotting = True,
                img_variations = not args.hidden,
                verbose=False,
                model = model,
                device = DEVICE
            ) 
        if args.wandb_project:
            wandb.log({"val/recons": wandb.Image(grid, caption=f"epoch{epoch_num+1:03d}")})
            plt.close()
        else:
            grid.savefig(os.path.join(args.output_dir_run, f'samples-val-epoch{epoch_num+1:03d}.png'))
            plt.show()
        vd_pipe = vd_pipe.to('cpu')
    return avg_epoch_loss, avg_epoch_loss_nce, avg_epoch_loss_mse, top1_accuracy, top5_accuracy


def train_new(model, train_dataloader, val_dataloader, args, DEVICE, vd_pipe):
    if args.wandb_project:
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = args.wandb_run_name if args.wandb_run_name else f"BM_B_{args.batch_size}_LR_{args.lr}_S_{args.seed}_E_{args.epochs}__{now}"
        wandb.init(project=args.wandb_project, name=run_name)
        wandb.config.update(vars(args))
    calculate_params(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    total_train_steps = len(train_dataloader)*args.epochs
    print(f"Total train steps: {total_train_steps}")
    
    if args.scheduler_type == "linear":
        # Warmup + Decay Scheduler
        warmup_steps = min(args.warmup_interval, total_train_steps)  # Ensure warmup_steps <= total_train_steps
        start_factor_w = args.warmup_lr / args.lr
        linear_decay_steps = total_train_steps - warmup_steps

        # Warmup and decay schedulers
        warmup_scheduler = LinearLR(optimizer, start_factor=start_factor_w, end_factor=1.0, total_iters=warmup_steps)
        if linear_decay_steps > 0:
            linear_decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=linear_decay_steps)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, linear_decay_scheduler], milestones=[warmup_steps])
        else:
            scheduler = warmup_scheduler  # Only warmup if no steps left for decay

        print("Using warmup + decay scheduler.")
    elif args.scheduler_type == "constant":
        # Constant learning rate scheduler
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_train_steps)
        print("Using constant learning rate scheduler.")
    
    if args.loss_func == "soft_clip_loss":
        loss_func = soft_clip_loss
    elif args.loss_func == "hard_clip_loss":
        loss_func = hard_clip_loss
    elif args.loss_func == "mse_loss":
        loss_func = mse_loss
    print("Loss Function in Train:", loss_func)
    print("Training started...")
    
    progress_bar = tqdm(range(args.epochs * len(train_dataloader)))

    iteration = 0
    collate_loss, collate_loss_val = [], []
    best_val_loss, patience_counter = 1e5, 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_model_path = os.path.join(output_dir, "best_model.pth")

    for epoch_num in range(args.epochs):
        epoch_loss, train_num_correct_top1, train_num_correct_top5, train_total = [], 0, 0, 0
        epoch_loss_nce = []
        epoch_loss_mse = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            meg = batch['meg'].to(DEVICE)
            embeddings = batch['image_embeddings'].to(DEVICE)
            # embeddings_normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
            embeddings_normalized = nn.functional.normalize(embeddings.flatten(1), dim=-1)
            
            ############################ FORWARD PASS ############################
            clip_head_out, mse_head_out = model(batch)
            clip_head_out = clip_head_out.to(DEVICE)
            mse_head_out = mse_head_out.to(DEVICE)
            # meg_normalized = clip_output / clip_output.norm(dim=1, keepdim=True)
            meg_normalized = nn.functional.normalize(clip_head_out.flatten(1), dim=-1)
            mse_out_normalized = nn.functional.normalize(mse_head_out.flatten(1), dim=-1)
            

            cos_similarities = torch.matmul(meg_normalized, embeddings_normalized.transpose(0, 1))
            labels = torch.arange(meg.size(0), dtype=torch.long, device=DEVICE)
            

            if loss_func in (soft_clip_loss, hard_clip_loss, mse_loss):
                loss_nce = loss_func(meg_normalized, embeddings_normalized)
            else:
                loss_nce = loss_func(cos_similarities, labels)
                
            loss_mse = mse_loss(mse_out_normalized, embeddings_normalized)
            
            if args.use_mse and args.use_clip:
                epoch_loss_nce.append(loss_nce.item())
                epoch_loss_mse.append(loss_mse.item())
                loss = (args.nce_mult * loss_nce) + (args.mse_mult * loss_mse)
            elif args.use_clip:
                epoch_loss_nce.append(loss_nce.item())
                loss = loss_nce
                loss_mse = torch.tensor(0.0)
            elif args.use_mse:
                epoch_loss_mse.append(loss_mse.item())
                loss = loss_mse # args.prior_mult * loss_prior
                loss_nce = torch.tensor(0.0)
            
            epoch_loss.append(loss.item())
            
            _, predicted_top1 = torch.max(cos_similarities, dim=1)
            train_num_correct_top1 += (predicted_top1 == labels).sum().item()

            _, predicted_top5 = cos_similarities.topk(5, dim=1)
            correct_top5 = predicted_top5.eq(labels.view(-1, 1).expand_as(predicted_top5))
            train_num_correct_top5 += correct_top5.sum().item()

            train_total += labels.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()
            iteration += 1
            progress_bar.update(1)

            if step % args.print_interval == 0:
                print(f"Epoch {epoch_num + 1}, Step {step + 1}: Train Loss = {loss.item():.6f}, Train NCE Loss = {loss_nce.item():.6f}, Train MSE Loss = {loss_mse.item():.6f}, LR = {scheduler.get_last_lr()[0]:.7f}")


        avg_tr_loss = sum(epoch_loss) / len(epoch_loss)
        avg_tr_loss_nce = sum(epoch_loss_nce) / len(epoch_loss_nce)
        if args.use_mse:
            avg_tr_loss_mse = sum(epoch_loss_mse) / len(epoch_loss_mse)
        else: avg_tr_loss_mse = 0.0
        collate_loss.append(avg_tr_loss)
        top1_accuracy_train = 100 * train_num_correct_top1 / train_total
        top5_accuracy_train = 100 * train_num_correct_top5 / train_total
        
        avg_val_loss, avg_val_loss_nce, avg_val_loss_mse, top1_acc, top5_acc = val_new(model, val_dataloader, loss_func, args, DEVICE, epoch_num, vd_pipe)
        collate_loss_val.append(avg_val_loss)
        print(
            f"[Epoch {epoch_num + 1}] Train Loss: {avg_tr_loss:.4f},  Train NCE Loss = {avg_tr_loss_nce:.4f}, Train MSE Loss = {avg_tr_loss_mse:.4f}, TRAIN Top-1 Acc: {top1_accuracy_train:.2f}%, TRAIN Top-5 Acc: {top5_accuracy_train:.2f}%\n"
            f"Val Loss: {avg_val_loss:.4f}, Val NCE Loss = {avg_val_loss_nce:.6f}, Val MSE Loss = {avg_val_loss_mse:.6f}, Top-1 Acc: {top1_acc:.6f}%, Top-5 Acc: {top5_acc:.6f}%"
        )

        if args.wandb_project:
            wandb.log({
                "epoch": epoch_num + 1,
                "step": iteration,
                "learning_rate": scheduler.get_last_lr()[0],
                "train_loss": avg_tr_loss,
                "train_loss_nce": avg_tr_loss_nce,
                "train_loss_mse": avg_tr_loss_mse,
                "train_top1_accuracy": top1_accuracy_train,
                "train_top5_accuracy": top5_accuracy_train,
                "val_loss": avg_val_loss,
                "val_loss_nce": avg_val_loss_nce,
                "val_loss_mse": avg_val_loss_mse,
                "val_top1_accuracy": top1_acc,
                "val_top5_accuracy": top5_acc
            })
        print("****** Prev Best Val Loss:", best_val_loss)
        print("****** Current Val Loss:", avg_val_loss)
        # Save best model based on Val Loss
        if avg_val_loss < best_val_loss:
            best_val_loss, patience_counter = avg_val_loss, 0
            print("Validation Loss improved !!!. Saving model...")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"*** No improvement in Val Loss for {patience_counter} epoch(s). ***")

        if args.early_stopping and patience_counter >= args.early_stopping:
            print(f"!!!! Early stopping triggered after {args.early_stopping} epochs of no improvement !!!!")
            break

        if (epoch_num + 1) % args.save_interval == 0:
            model_path = os.path.join(output_dir, f"model_epoch_{epoch_num + 1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    print(f"Training complete. Best model saved at {best_model_path}.")

    progress_bar.close()
    return model, collate_loss, collate_loss_val, best_model_path



def test_new(model, test_dataloader, best_model_path, args, DEVICE):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(args)
    if args.loss_func == "soft_clip_loss":
        loss_func = soft_clip_loss
    elif args.loss_func == "hard_clip_loss":
        loss_func = hard_clip_loss
    elif args.loss_func == "mse_loss":
        loss_func = mse_loss
    print("Loss Function in Test:", loss_func)
    avg_test_loss, top1_accuracy, top5_accuracy = val_new(model, test_dataloader, loss_func, DEVICE)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Top-1 Accuracy: {top1_accuracy:.2f}%, Test Top-5 Accuracy: {top5_accuracy:.2f}%")

    if args.wandb_project:
        wandb.log({
            "test_loss": avg_test_loss,
            "test_top1_accuracy": top1_accuracy,
            "test_top5_accuracy": top5_accuracy,
        })
    return avg_test_loss, top1_accuracy, top5_accuracy



#####################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
@torch.no_grad()
def reconstruction_new(
    # image, 
    batch,
    clip_extractor = None,
    unet=None, 
    vae=None, 
    noise_scheduler=None,
    voxel2clip_cls=None,
    # diffusion_priors=None,
    text_token = None,
    img_lowlevel = None,
    num_inference_steps = 50,
    recons_per_sample = 1,
    guidance_scale = 7.5,
    img2img_strength = .85,
    timesteps_prior = 100,
    seed = 42,
    retrieve=False,
    plotting=True,
    verbose=False,
    img_variations=False,
    n_samples_save=1,
    # num_retrieved=16,
    device=device,# PASS this,
    model= None # SimpleCONV
):
    assert n_samples_save==1, "n_samples_save must = 1. Function must be called one image at a time"
    
    brain_recons = None
    
    # print({key: type(value) for key, value in batch.items()})
    batch = {key: slice_batch(value, n_samples_save) for key, value in batch.items()}
    # image=image[:n_samples_save] # CHANGE TO IMAGE FROM PATH?

    # Get the image path from the batch metadata
    img_path = batch['metadata']['image_path'][0]  # Assumes `image_path` exists in metadata

    print(f"Image path: {img_path}")
    # Define the base path to the image directory
    base_path = "/scratch/dm5927/cv_project/THINGS/osfstorage-archive/object_images/"

    # Combine the base path and relative image path
    full_image_path = os.path.join(base_path, img_path)

    # Open the image
    image = Image.open(full_image_path).convert('RGB')  # Ensure 'RGB' mode

    # Resize the image
    image = image.resize((150, 150))  # Resizes to 150x150 pixels

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    clip_embeddings, brain_mse_embeddings0  = model(batch)
    brain_mse_embeddings0 = brain_mse_embeddings0.contiguous().view(len(batch['meg']),-1,768).to(device)
    brain_clip_embeddings = brain_mse_embeddings0
    
    if voxel2clip_cls is not None: #WONT GO INTO HERE
        _, cls_embeddings = voxel2clip_cls(batch['meg'].to(device).float())
    else:
        cls_embeddings = clip_embeddings
    if verbose: print("cls_embeddings.",cls_embeddings.shape)
    
    if recons_per_sample > 0:
        if not img_variations:
            for samp in range(len(brain_clip_embeddings)):
                brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        else:
            brain_clip_embeddings = brain_clip_embeddings.unsqueeze(1)
        
        input_embedding = brain_clip_embeddings#.repeat(recons_per_sample, 1, 1)
        if verbose: print("input_embedding",input_embedding.shape)

        if text_token is not None:
            prompt_embeds = text_token.repeat(recons_per_sample, 1, 1)
        else:
            prompt_embeds = torch.zeros(len(input_embedding),77,768)
        if verbose: print("prompt!",prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)

        # dual_prompt_embeddings
        if not img_variations:
            input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        # 5b. Prepare latent variables
        batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
        shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        
        # DOES NOT GO Into here
        if img_lowlevel is not None: # use img_lowlevel for img2img initialization
            init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(batch_size)
            
            if verbose: print("img_lowlevel", img_lowlevel.shape)
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                                generator=generator, dtype=input_embedding.dtype)
            init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                  generator=generator, dtype=input_embedding.dtype)
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            if verbose: print("latent_model_input", latent_model_input.shape)
            if verbose: print("input_embedding", input_embedding.shape)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # TODO:
                # noise_pred = dynamic_cfg(noise_pred_uncond, noise_pred_text, guidance_scale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        recons = decode_latents(latents,vae).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose: print("brain_recons",brain_recons.shape)
                    
    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)
    
    if retrieve==False:
        v2c_reference_out = nn.functional.normalize(clip_embeddings.contiguous().view(len(clip_embeddings),-1),dim=-1)
        sims=[]
        for im in range(recons_per_sample): 
            # currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(clip_embeddings.device).to(clip_embeddings.dtype)
            # currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
            currecon = nn.functional.normalize(batch['image_embeddings'].flatten(),dim=-1).to(clip_embeddings.device)
            cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
            sims.append(cursim.item())
        if verbose: print(sims)
        best_picks[0] = int(np.nanargmax(sims))   
        if verbose: print(best_picks)
    else:  # WON'T GO INTO HERE
        v2c_reference_out = nn.functional.normalize(clip_embeddings.contiguous().view(len(clip_embeddings),-1),dim=-1)
        retrieved_clips = clip_extractor.embed_image(torch.Tensor(image_retrieved).to(device)).float()
        sims=[]
        for ii,im in enumerate(retrieved_clips):
            currecon = nn.functional.normalize(im.flatten()[None],dim=-1)
            if verbose: print(v2c_reference_out.shape, currecon.shape)
            cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
            sims.append(cursim.item())
        if verbose: print(sims)
        best_picks[0] = int(np.nanargmax(sims)) 
        if verbose: print(best_picks)
        recon_img = image_retrieved[best_picks[0]]
    
    if recons_per_sample==0 and retrieve: # WON'T GO INTO HERE
        recon_is_laion = True
        recons_per_sample = 1 # brain reconstruction will simply be the LAION nearest neighbor
    else:
        recon_is_laion = False
                    
    img2img_samples = 0 if img_lowlevel is None else 1
    laion_samples = 1 if retrieve else 0
    num_xaxis_subplots = 1+img2img_samples+laion_samples+recons_per_sample
    if plotting:
        fig, ax = plt.subplots(n_samples_save, num_xaxis_subplots, 
                           figsize=(num_xaxis_subplots*5,6*n_samples_save),facecolor=(1, 1, 1))
    else:
        fig = None
        recon_img = None
    
    im = 0
    if plotting:
        ax[0].set_title(f"Original Image")
        ax[0].imshow(image)
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0,1)))
    for ii,i in enumerate(range(num_xaxis_subplots-laion_samples-recons_per_sample,num_xaxis_subplots-laion_samples)):
        recon = brain_recons[im][ii]
        if recon_is_laion:
            recon = brain_recons[best_picks[0]]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction",fontweight='bold')
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    # if plotting:
    #     if retrieve and not recon_is_laion:
    #         ax[-1].set_title(f"LAION5b top neighbor")
    #         ax[-1].imshow(torch_to_Image(image_retrieved0))
    #     for i in range(num_xaxis_subplots):
    #         ax[i].axis('off')
    
    return fig, brain_recons, best_picks, recon_img



#FOR RECONSTRUCTION, HELPER FUNCTIONS
def slice_batch(data, n_samples_save):
    """
    Recursively slices tensors, lists, or tuples in a dictionary.
    """
    if isinstance(data, (torch.Tensor, list, tuple)):
        return data[n_samples_save-1:n_samples_save]  # Slice tensors, lists, or tuples
    elif isinstance(data, dict):
        return {key: slice_batch(value, n_samples_save) for key, value in data.items()}  # Recurse for dicts
    else:
        return data
    

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')
    
def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]
    
def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity


