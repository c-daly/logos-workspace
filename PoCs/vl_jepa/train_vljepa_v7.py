"""VL-JEPA v7: cosine-focused fine-tune from v6 best. Goal: cosine >= 0.70."""
import argparse, os
import h5py, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim*2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dim*2, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.net(x)


class ResidualTranslator(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, hidden_dim=1024, num_blocks=4, dropout=0.05):
        super().__init__()
        self.proj_in  = nn.Linear(input_dim, hidden_dim)
        self.blocks   = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.proj_out = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        if x.dim() == 3: x = x.mean(dim=1)
        return F.normalize(self.proj_out(self.blocks(self.proj_in(x))), dim=-1)


def compute_loss(proj, tgt, batch_indices, temp=0.07):
    # Heavy cosine weight + light InfoNCE — no MSE
    cos_loss = 1.0 - (proj * tgt).sum(-1).mean()
    B = proj.shape[0]
    sim = proj @ tgt.T / temp
    vi = torch.tensor(batch_indices, device=proj.device)
    fn = (vi.unsqueeze(1) == vi.unsqueeze(0)) & ~torch.eye(B, dtype=torch.bool, device=proj.device)
    sim[fn] = -1e9
    lbl = torch.arange(B, device=proj.device)
    nce = (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.T, lbl)) / 2
    return 0.8 * cos_loss + 0.2 * nce


@torch.no_grad()
def eval_retrieval(model, jepa_val, clip_val, device, bs=256):
    model.eval()
    projs = [model(jepa_val[i:i+bs].to(device)).cpu() for i in range(0, len(jepa_val), bs)]
    proj = torch.cat(projs)
    clip_n = F.normalize(clip_val, dim=-1)
    cosines = (proj * clip_n).sum(-1).mean().item()
    sim = proj @ clip_n.T
    lbl = torch.arange(len(proj))
    r1 = (sim.argmax(1) == lbl).float().mean().item()
    r5 = (sim.topk(5,1).indices == lbl.unsqueeze(1)).any(1).float().mean().item()
    model.train()
    return r1, r5, cosines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",             required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints_v7")
    p.add_argument("--warm-start",     default=None)
    p.add_argument("--lr",   type=float, default=5e-5)
    p.add_argument("--batch",type=int,   default=512)
    p.add_argument("--epochs",type=int,  default=300)
    p.add_argument("--patience",type=int,default=60)
    p.add_argument("--temp", type=float, default=0.07)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading embeddings...", flush=True)
    with h5py.File(args.h5, "r") as f:
        jepa_all  = torch.tensor(f["jepa_embeddings"][:],       dtype=torch.float32)
        clip_text = torch.tensor(f["clip_text_embeddings"][:],  dtype=torch.float32)
        clip_img  = torch.tensor(f["clip_image_embeddings"][:], dtype=torch.float32)

    clip_text_mean = F.normalize(clip_text.mean(1), dim=-1)
    clip_img_mean  = F.normalize(clip_img.mean(1),  dim=-1)
    clip_target    = F.normalize(0.7*clip_text_mean + 0.3*clip_img_mean, dim=-1)

    N = len(jepa_all)
    perm    = np.random.default_rng(42).permutation(N)
    n_val   = max(1, int(0.2*N))
    val_idx = sorted(perm[:n_val].tolist())
    trn_idx = sorted(perm[n_val:].tolist())
    print(f"Train: {len(trn_idx)}, Val: {len(val_idx)}", flush=True)

    jepa_trn = jepa_all[trn_idx]
    jepa_val = jepa_all[val_idx]
    clip_trn = clip_target[trn_idx]
    clip_val = clip_target[val_idx]

    model = ResidualTranslator().to(device)
    if args.warm_start:
        ckpt = torch.load(args.warm_start, map_location=device)
        key = "model_state_dict" if "model_state_dict" in ckpt else "model"
        model.load_state_dict(ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt)
        print(f"Warm-started from {args.warm_start}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_cosine = 0.0
    no_improve  = 0

    for epoch in range(args.epochs):
        model.train()
        perm_e = torch.randperm(len(jepa_trn))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(jepa_trn), args.batch):
            idx = perm_e[i:i+args.batch]
            xb  = jepa_trn[idx].to(device)
            tb  = clip_trn[idx].to(device)
            proj = model(xb)
            loss = compute_loss(proj, tb, idx.tolist(), args.temp)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        sched.step()

        r1, r5, cosine = eval_retrieval(model, jepa_val, clip_val, device)
        avg_loss = total_loss / max(1, n_batches)
        lr_disp = opt.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | R@1={r1:.4f} | R@5={r5:.4f} | cos={cosine:.4f} | lr={lr_disp:.2e}", flush=True)

        if cosine > best_cosine:
            best_cosine = cosine
            no_improve = 0
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "cosine": cosine, "r5": r5, "r1": r1},
                       os.path.join(args.checkpoint_dir, "best_vljepa_v7.pt"))
            print(f"  -> New best cosine={cosine:.4f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {epoch+1}", flush=True)
                break

    print(f"\nFinal best cosine={best_cosine:.4f}", flush=True)


if __name__ == "__main__":
    main()
