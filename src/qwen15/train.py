from unsloth import FastLanguageModel
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
# Removed PEFT imports as they are no longer needed
from trl import SFTTrainer, SFTConfig
import gc
from transformers import TrainerCallback
import wandb
import re
import bitsandbytes as bnb

DATASET = "nvidia/OpenMathReasoning"
DATASET_SPLIT = "cot"

class ExpertUsageTracker:
    def __init__(self, num_experts, top_k=2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.counts = torch.zeros(num_experts, device="cuda")

    def hook_fn(self, module, input, output):
        # Qwen2 router returns (router_logits, selected_indices)
        # We want the logits to see which experts were chosen
        if isinstance(output, (tuple, list)):
            logits = output[0] 
        else:
            logits = output

        if isinstance(logits, torch.Tensor):
            # Logits shape is [batch_size * sequence_length, num_experts]
            _, indices = torch.topk(logits, self.top_k, dim=-1)
            flat_indices = indices.flatten()
            batch_counts = torch.bincount(flat_indices, minlength=self.num_experts)
            self.counts += batch_counts[:self.num_experts]

    def get_probabilities(self):
        total = self.counts.sum()
        if total == 0: return torch.zeros(self.num_experts).cpu().numpy()
        return (self.counts / total).cpu().detach().numpy()

def format_sample(sample):
    return {"text": f"Problem: {sample['problem']}\n\nSolution: {sample['generated_solution']}"}

def get_expert_tensors(model, state):
    expert_bank = {}
    for name, module in model.named_modules():
        if name.endswith("mlp.experts"):
            m = re.search(r"layers\.(\d+)", name)
            layer_id = m.group(1) if m else "0"
            
            for proj_name in ["gate_up_proj", "down_proj"]:
                proj_attr = getattr(module, proj_name, None)
                
                if proj_attr is not None:
                    if hasattr(proj_attr, "weight"):
                        w = proj_attr.weight.data
                    else:
                        w = proj_attr.data
                    
                    # Ensure we are working with the expected 3D shape: [E, Out, In]
                    # w[i] is expert i
                    try:
                        expert_bank[f"layer_{layer_id}_{proj_name}"] = [
                            w[i].flatten().float().cpu() for i in range(w.shape[0])
                        ]
                    except Exception as e:
                        continue        
    return expert_bank

# def compute_global_expert_diversity(expert_bank):
#     all_off_diag = []
#     for layer_proj, vectors in expert_bank.items():
#         if len(vectors) < 2: continue
#         try:
#             W = torch.stack(vectors)  # [E, D]
#             W = F.normalize(W, dim=1)
#             sim = W @ W.T
#             mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
#             all_off_diag.append(sim[mask])
#         except Exception: continue

#     if not all_off_diag: return None
#     all_off_diag = torch.cat(all_off_diag)
#     return {
#         "mean_cosine_similarity": all_off_diag.mean().item(),
#         "max_cosine_similarity": all_off_diag.max().item(),
#         "min_cosine_similarity": all_off_diag.min().item(),
#         "std_cosine_similarity": all_off_diag.std().item(),
#     }
def compute_global_expert_diversity(expert_bank):
    all_off_diag = []
    num_parts = 4  # The number of distinct matrices in one "copy"
    
    for layer_proj, vectors in expert_bank.items():
        if len(vectors) < num_parts: continue
        
        try:
            # W shape: [num_experts, D], e.g., [60, D]
            W = torch.stack(vectors) 
            num_experts = W.shape[0]
            
            # We want to compare experts that are at the same position in their groups
            # i.e., indices: [0, 4, 8, ...], [1, 5, 9, ...], etc.
            for part_idx in range(num_parts):
                # Select experts 0, 4, 8... then 1, 5, 9...
                indices = torch.arange(part_idx, num_experts, num_parts)
                part_W = W[indices]  # Shape: [num_copies, D], e.g., [15, D]
                
                if part_W.shape[0] < 2: continue
                
                # Compare these parallel experts against each other
                part_W = F.normalize(part_W, dim=1)
                sim = part_W @ part_W.T # [15, 15]
                
                # Mask out diagonal
                mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
                all_off_diag.append(sim[mask])
                
        except Exception:
            continue

    if not all_off_diag: return None
    
    all_off_diag = torch.cat(all_off_diag)
    
    return {
        "mean_cosine_similarity": all_off_diag.mean().item(),
        "max_cosine_similarity": all_off_diag.max().item(),
        "min_cosine_similarity": all_off_diag.min().item(),
        "std_cosine_similarity": all_off_diag.std().item(),
    }


# class ExpertLoggingCallback(TrainerCallback):
#     def __init__(self, tracker):
#         self.tracker = tracker

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if state.is_world_process_zero:
#             probs = self.tracker.get_probabilities()
#             data = [[i, p] for i, p in enumerate(probs)]
#             table = wandb.Table(data=data, columns=["expert_id", "usage_fraction"])
            
#             wandb.log({
#                 "expert_usage_dist": wandb.plot.bar(table, "expert_id", "usage_fraction", title="Expert Usage Distribution"),
#                 "max_expert_load": probs.max(),
#                 "min_expert_load": probs.min()
#             }, step=state.global_step, commit=False)

# class ExpertDiversityCallback(TrainerCallback):
#     def __init__(self, model, every_n_steps=10):
#         self.model = model
#         self.every_n_steps = every_n_steps

#     def on_step_end(self, args, state, control, **kwargs):
#         if state.global_step % self.every_n_steps != 0: return
#         expert_bank = get_expert_tensors(self.model, state)
#         stats = compute_global_expert_diversity(expert_bank)
#         if stats:
#             wandb.log({f"expert_diversity/{k}": v for k, v in stats.items()}, 
#                       step=state.global_step, commit=False)

class ExpertCallback(TrainerCallback):
    def __init__(self, tracker, model, every_n_steps=10):
        self.tracker = tracker
        self.model = model
        self.every_n_steps = every_n_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            probs = self.tracker.get_probabilities()
            data = [[i, p] for i, p in enumerate(probs)]
            table = wandb.Table(data=data, columns=["expert_id", "usage_fraction"])
            
            wandb.log({
                "expert_usage_dist": wandb.plot.bar(table, "expert_id", "usage_fraction", title="Expert Usage Distribution"),
                "max_expert_load": probs.max(),
                "min_expert_load": probs.min()
            }, step=state.global_step, commit=False)

        
        if state.global_step % self.every_n_steps != 0: return
        expert_bank = get_expert_tensors(self.model, state)
        stats = compute_global_expert_diversity(expert_bank)
        if stats:
            wandb.log({f"expert_diversity/{k}": v for k, v in stats.items()}, 
                      step=state.global_step, commit=False)

def main():
    parser = argparse.ArgumentParser(description="Full fine-tune upcycled MoE model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", default="moe-full-ft")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1) # Lowered for full FT memory
    parser.add_argument("--lr", type=float, default=2e-5)    # Lower LR usually better for full FT
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output", default="/tmp/moe-checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dataset-seed", type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.seq_len,
        load_in_4bit = False, 
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model.requires_grad_(True)
    model.requires_grad_(False)

    # Unfreeze only experts and gates
    for name, param in model.named_parameters():
        if "experts" in name or "gate" in name:
            param.requires_grad = True
    # print("\n--- MOE LAYER DISCOVERY ---")
    # for name, module in model.named_modules():
    #     # Find the Gates/Routers
    #     if "gate" in name.lower() or "router" in name.lower():
    #         print(f"FOUND GATE: {name} | Type: {type(module)}")
            
    #     # Find the Experts
    #     if "experts" in name.lower() and len(list(module.children())) == 0:
    #         # We only print leaf nodes (the actual linear layers)
    #         print(f"FOUND EXPERT WEIGHT: {name} | Shape: {module.weight.shape if hasattr(module, 'weight') else 'N/A'}")
    
    # Enable gradient checkpointing to save VRAM during full training
    model.gradient_checkpointing_enable()

    num_experts = getattr(model.config, "num_experts", 60)
    top_k = getattr(model.config, "num_experts_per_tok", 2)
    tracker = ExpertUsageTracker(num_experts, top_k=top_k)
    
    # for name, module in model.named_modules():
    #     if name.endswith(".gate"):
    #         module.register_forward_hook(tracker.hook_fn)

    # UPDATED CODE
    hooks_count = 0
    for name, module in model.named_modules():
        # Match the specific Top-K router discovered in your logs
        if name.endswith(".mlp.gate"):
            module.register_forward_hook(tracker.hook_fn)
            hooks_count += 1
            # print(f"✅ Hook attached to Router: {name}")

    if hooks_count == 0:
        print("❌ ERROR: No MoE routers found! Check the naming hierarchy again.")
    else:
        print(f"Successfully attached {hooks_count} hooks to MoE routers.")
    
    print("Model loaded for Full Fine-Tuning.")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    ds = load_dataset(DATASET, split=DATASET_SPLIT, streaming=True)
    ds = ds.shuffle(seed=args.dataset_seed, buffer_size=1000).map(format_sample)

    training_args = SFTConfig(
        output_dir=f"{args.output}/{args.run_name}",
        run_name=args.run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8, # Increased to compensate for lower batch size
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_steps=args.max_steps,
        optim = "adamw_8bit", # Highly recommended for full MoE fine-tuning
        bf16 = torch.cuda.is_bf16_supported(),
        fp16 = not torch.cuda.is_bf16_supported(),
        report_to="none" if args.no_wandb else "wandb",
        dataset_text_field="text",
        max_length=args.seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=training_args,
        callbacks=[
            ExpertCallback(tracker, model, every_n_steps=10),
            # ExpertLoggingCallback(tracker),
            # ExpertDiversityCallback(model, every_n_steps=10),
        ]
    )

    trainer.train()
    trainer.save_model(f"{args.output}/{args.run_name}/final")

if __name__ == "__main__":
    main()