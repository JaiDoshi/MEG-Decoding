import torch
import os
import wandb
import time
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, LambdaLR
# from transformers import AdamW, get_scheduler
from tqdm import tqdm
import numpy as np

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    # print(clip_clip)
    brain_clip = (preds @ targs.T)/temp
    # print(brain_clip)
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def mse_loss(preds, targs):
    preds_scaled = preds * 10
    targs_scaled = targs * 10
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

    return 

def val_modified(conv_model, vit_model, dataloader, loss_func, DEVICE, args):
    conv_model.eval()
    vit_model.eval()
    epoch_loss, num_correct_top1, num_correct_top5, total = [], 0, 0, 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            # print(f"Batch {i + 1} of {len(dataloader)}")

            embeddings = batch['image_embeddings'].to(DEVICE)
            embeddings_normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
            
            attn_output = conv_model(batch).to(DEVICE)
            vit_output = vit_model(attn_output)[0].to(DEVICE)
            brain_normalized = vit_output / vit_output.norm(dim=1, keepdim=True)

            cos_similarities = torch.matmul(brain_normalized, embeddings_normalized.transpose(0, 1))
            labels = torch.arange(dataloader.batch_size, dtype=torch.long, device=DEVICE)

        
            if loss_func in (soft_clip_loss, hard_clip_loss, mse_loss):
                loss = loss_func(brain_normalized, embeddings_normalized)
                
            else: 
                loss = loss_func(cos_similarities, labels)

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


def train_modified(conv_model, vit_model, train_dataloader, val_dataloader, args, DEVICE):
    if args.wandb_project:
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        
    if args.wandb_project and args.wandb_run_name:
        run_name = args.wandb_run_name
        wandb.init(project=args.wandb_project, name=run_name)
        wandb.config.update(vars(args))
    else:
        raise ValueError("Please provide a run name for wandb")
    
    print("Parameter counts in Conv Model:")
    calculate_params(conv_model)
    print("Parameter counts in Vision Transformer Model:")
    calculate_params(vit_model)
    
    
    total_train_steps = len(train_dataloader)*args.epochs
    print(f"Total train steps: {total_train_steps}")

    warmup_steps = args.warmup_interval
    # The desired final LR at warmup_steps is args.lr
    d_model = args.hidden_size 
    
    # Compute the scaling factor
    base_rate_at_warmup = (d_model ** (-0.5)) * (warmup_steps ** (-0.5))
    scale_factor = args.lr / base_rate_at_warmup

    optimizer = optim.Adam(
        list(conv_model.parameters()) + list(vit_model.parameters()),
        lr=args.lr, betas=(0.9, 0.98), eps=1e-9
    )
    
    # Define the lambda function for LR
    def lr_lambda(step):
        # Ensure step is at least 1 to avoid division by zero
        step = max(step, 1)
        # Apply the formula:
        # lrate = scale_factor * d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
        arg1 = step ** (-0.5)
        arg2 = step * (warmup_steps ** (-1.5))
        return scale_factor * (d_model ** (-0.5)) * min(arg1, arg2)

    # Create a LambdaLR scheduler. Note that by default, step count starts at 1.
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    
    # warmup_steps = args.warmup_interval
    # warmup_lr = args.warmup_lr
    # start_factor_w = args.warmup_lr/ args.lr
    # num_train_batches = len(train_dataloader)
    # optimizer = optim.Adam(
    #     list(conv_model.parameters()) + list(vit_model.parameters()),
    #     lr=args.lr, betas=(0.9, 0.98), eps=1e-09
    #     )   
    # # optimizer_conv = optim.Adam(conv_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # # optimizer_vit = optim.Adam(vit_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # print("Warm up learning rate:", warmup_lr)
    # print("Warm up steps:", warmup_steps)
    # warmup_scheduler = LinearLR(optimizer, start_factor=start_factor_w, end_factor=1.0, total_iters=warmup_steps)
    # linear_decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_train_steps - warmup_steps)
    
    # # Combine warmup and linear decay using SequentialLR
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, linear_decay_scheduler], milestones=[warmup_steps])
    
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

    conv_best_model_path = os.path.join(output_dir, "conv_best_model.pth")
    vit_best_model_path = os.path.join(output_dir, "vit_best_model.pth")

    for epoch_num in range(args.epochs):
        epoch_loss, train_num_correct_top1, train_num_correct_top5, train_total = [], 0, 0, 0
        conv_model.train()
        vit_model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # optimizer_conv.zero_grad()
            # optimizer_vit.zero_grad()

            embeddings = batch['image_embeddings'].to(DEVICE)
            embeddings_normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
            
            attn_output = conv_model(batch).to(DEVICE)
            vit_output = vit_model(attn_output)[0].to(DEVICE)
            brain_normalized = vit_output / vit_output.norm(dim=1, keepdim=True)

            cos_similarities = torch.matmul(brain_normalized, embeddings_normalized.transpose(0, 1))
            labels = torch.arange(train_dataloader.batch_size, dtype=torch.long, device=DEVICE)

            if loss_func in (soft_clip_loss, hard_clip_loss, mse_loss):
                loss = loss_func(brain_normalized, embeddings_normalized)
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
            # optimizer_conv.step()
            # optimizer_vit.step()
            scheduler.step()
            iteration += 1
            progress_bar.update(1)

            if step % args.print_interval == 0:
                print(f"Epoch {epoch_num + 1}, Step {step + 1}: Train Loss = {loss.item():.4f}, LR = {scheduler.get_last_lr()[0]:.7f}")


        avg_tr_loss = sum(epoch_loss) / len(epoch_loss)
        collate_loss.append(avg_tr_loss)
        top1_accuracy_train = 100 * train_num_correct_top1 / train_total
        top5_accuracy_train = 100 * train_num_correct_top5 / train_total
        
        avg_val_loss, top1_acc, top5_acc = val_modified(conv_model, vit_model, val_dataloader, loss_func, DEVICE, args)
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
            print("Validation Top-5 accuracy improved !!!. Saving models...")
            torch.save(conv_model.state_dict(), conv_best_model_path)
            torch.save(vit_model.state_dict(), vit_best_model_path)
        else:
            patience_counter += 1
            print(f"*** No improvement in Top-5 accuracy for {patience_counter} epoch(s). ***")

        if args.early_stopping and patience_counter >= args.early_stopping:
            print(f"!!!! Early stopping triggered after {args.early_stopping} epochs of no improvement !!!!")
            break

        if (epoch_num + 1) % args.save_interval == 0:
            conv_model_path = os.path.join(output_dir, f"conv_model_epoch_{epoch_num + 1}.pth")
            vit_model_path = os.path.join(output_dir, f"vit_model_epoch_{epoch_num + 1}.pth")
            torch.save(conv_model.state_dict(), conv_model_path)
            torch.save(vit_model.state_dict(), vit_model_path)
            print(f"Models saved to {conv_model_path} and {vit_model_path}")

    print(f"Training complete. Conv Best model saved at {conv_best_model_path}. ViT Best model saved at {vit_best_model_path}")

    progress_bar.close()
    return conv_model, vit_model, collate_loss, collate_loss_val, conv_best_model_path, vit_best_model_path



def test_model(conv_model, vit_model, test_dataloader, conv_best_model_path, vit_best_model_path, args, DEVICE):
    state_dict1 = torch.load(conv_best_model_path, map_location=DEVICE)
    state_dict2 = torch.load(vit_best_model_path, map_location=DEVICE)
    conv_model.load_state_dict(state_dict1)
    vit_model.load_state_dict(state_dict2)
    conv_model.to(DEVICE)
    vit_model.to(DEVICE)
    conv_model.eval()
    vit_model.eval()
    print(args)
    if args.loss_func == "soft_clip_loss":
        loss_func = soft_clip_loss
    elif args.loss_func == "hard_clip_loss":
        loss_func = hard_clip_loss
    elif args.loss_func == "mse_loss":
        loss_func = mse_loss
    print("Loss Function in Test:", loss_func)
    avg_test_loss, top1_accuracy, top5_accuracy = val_modified(conv_model, vit_model, test_dataloader, loss_func, DEVICE, args)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Top-1 Accuracy: {top1_accuracy:.2f}%, Test Top-5 Accuracy: {top5_accuracy:.2f}%")

    if args.wandb_project:
        wandb.log({
            "test_loss": avg_test_loss,
            "test_top1_accuracy": top1_accuracy,
            "test_top5_accuracy": top5_accuracy,
        })
    return avg_test_loss, top1_accuracy, top5_accuracy


# def test_modified(conv_model, vit_model, dataloader, loss_func, DEVICE):
#     conv_model.eval()
#     vit_model.eval()
#     epoch_loss, num_correct_top1, num_correct_top5, total = [], 0, 0, 0
#     with torch.no_grad():
#         for i, batch in tqdm(enumerate(dataloader)):
#             # print(f"Batch {i + 1} of {len(dataloader)}")
#             meg = batch['meg'].to(DEVICE)
#             embeddings = batch['image_embeddings'].to(DEVICE)
#             embeddings_normalized = embeddings / embeddings.norm(dim=1, keepdim=True)
#             # print("Norm of Image Emb:", embeddings_normalized.norm()) 
#             labels = torch.arange(meg.size(0), dtype=torch.long, device=DEVICE)

#             clip_output = model(batch).to(DEVICE)
#             meg_normalized = clip_output / clip_output.norm(dim=1, keepdim=True)
#             # print("Norm of MEG:", meg_normalized.norm())    
            
#             cos_similarities = torch.matmul(meg_normalized, embeddings_normalized.transpose(0, 1))
#             if loss_func in (soft_clip_loss, hard_clip_loss, mse_loss):
#                 loss = loss_func(meg_normalized, embeddings_normalized)
                
#             else: loss = loss_func(cos_similarities, labels)
#             epoch_loss.append(loss.item())

#             _, predicted_top1 = torch.max(cos_similarities, dim=1)
#             num_correct_top1 += (predicted_top1 == labels).sum().item()

#             _, predicted_top5 = cos_similarities.topk(5, dim=1)
#             correct_top5 = predicted_top5.eq(labels.view(-1, 1).expand_as(predicted_top5))
#             num_correct_top5 += correct_top5.sum().item()

#             total += labels.size(0)

#     avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
#     top1_accuracy = 100 * num_correct_top1 / total
#     top5_accuracy = 100 * num_correct_top5 / total
#     return avg_epoch_loss, top1_accuracy, top5_accuracy