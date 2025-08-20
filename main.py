import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from FV_PIFNO import Net2d
from Diffusion import PhysParams, diffusion_residual

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def build_dataloaders(discret_npz, perm_path: str, sample_nums: int, train_samples: int,
                      batch_size: int, shuffle: bool, device):
    size = int(discret_npz["size"])
    data_max = float(discret_npz["data_max"])
    grid_data = discret_npz["grid_data"].astype(np.float32)  # (N,2)

    perm = np.load(perm_path) * 1e-12                        # (S, N) or (S, H*W)
    perm = perm.reshape(sample_nums, size, size, 1)
    permscale = perm.max() - perm.min()
    perm_norm = perm / permscale

    coords = grid_data.reshape(size, size, 2).astype(np.float32) / data_max
    coords = coords[None, ...]                               # (1,H,W,2)
    coords_train = np.repeat(coords, train_samples, axis=0)  # (B,H,W,2)

    train_input = np.concatenate([perm_norm[:train_samples], coords_train], axis=-1)  # (B,H,W,3)
    train_input = torch.from_numpy(train_input.astype(np.float32))                    # CPU tensor

    dataset = TensorDataset(train_input, torch.zeros(train_input.shape[0]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, permscale

def train(args):
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    disc_path = Path(args.discret_path)
    assert disc_path.exists(), f"Discretization file not found: {disc_path}"
    discret_npz = np.load(disc_path)

    loader, permscale = build_dataloaders(
        discret_npz, args.perm_path, args.sample_nums, args.train_samples,
        args.batch_size, args.shuffle, device
    )

    size = int(discret_npz["size"])
    input_dim = 3
    model = Net2d(modes=args.modes, width=args.width, size=size, input_dim=input_dim).to(device)
    model.train()

    phys = PhysParams(discret_npz, device=device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        running = 0.0
        for (data_input, _) in loader:
            optimizer.zero_grad()

            # forward (B,H,W) -> (N,B)
            Pressure = model(data_input.to(device))                  # (B,H,W)
            B = Pressure.shape[0]
            N = size * size
            Pressure = Pressure.contiguous().view(B, N).permute(1, 0)  # (N,B)

            #  Restore physics scale
            km = data_input[:, :, :, 0].contiguous().view(B, N).permute(1, 0).to(device) * permscale  # (N,B)

            diff = diffusion_residual(km, Pressure, phys, mu_scalar=args.mu)

            loss = criterion(diff, torch.zeros_like(diff))
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_loss = running / len(loader)
        # print(f"Epoch {ep}/{args.epochs} | Train Loss: {avg_loss:.6f}")
        print(f"Epoch {ep}/{args.epochs} | Train Loss: {avg_loss}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    print(f"Saved final model to {str((out_dir / 'model_final.pt').resolve())}")

def main():
    parser = argparse.ArgumentParser()
    # load
    parser.add_argument("--discret_path", type=str, default="data/2D_meshgrid.npz")
    parser.add_argument("--perm_path", type=str, default="data/permeability_2D_1_2_Field average 16.29-31.37, 22.73, 2.53; Contrast ratio 5.58-57.75, 14.58, 5.41.npy")
    parser.add_argument("--sample_nums", type=int, default=1000)
    parser.add_argument("--train_samples", type=int, default=100)

    # train
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--mu", type=float, default=1e-9)

    # output
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=888888)
    parser.add_argument("--out_dir", type=str, default="outputs")

    parser.add_argument("--shuffle", action="store_true", default=True)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
