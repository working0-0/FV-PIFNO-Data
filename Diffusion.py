import torch
from torch_geometric.utils import to_dense_adj

class PhysParams:
    def __init__(self, npz, device):
        self.size = int(npz["size"])
        self.N = self.size * self.size
        self.data_max = float(npz["data_max"])
        self.p_l = float(npz["p_l"])
        self.p_r = float(npz["p_r"])
        self.edge_index = torch.as_tensor(npz["edge_index"], dtype=torch.long, device=device)  # (2,E)
        self.w_1 = float(npz["w_1"])
        self.w_2 = float(npz["w_2"])
        self.conn_d1 = float(npz["conn_d1"])
        self.conn_d2 = float(npz["conn_d2"])
        self.connList_areas = torch.as_tensor(npz["connList_areas"], dtype=torch.float32, device=device)  # (N,1)
        self.prd_l_wellindex = torch.as_tensor(npz["prd_l_wellindex"], dtype=torch.long, device=device)   # (size,)
        self.prd_r_wellindex = torch.as_tensor(npz["prd_r_wellindex"], dtype=torch.long, device=device)   # (size,)

        self.ones = torch.ones((self.N, 1), dtype=torch.float32, device=device)
        self.eye = torch.eye(self.N, self.N, dtype=torch.float32, device=device)

def diffusion_residual(km, P, phys: PhysParams, mu_scalar: float):
    # km: (N,B), P: (N,B)
    aa = phys.edge_index[0]  # (E,)
    bb = phys.edge_index[1]  # (E,)

    # k_ij: (E,B)
    # (E,1) <- (N,1)[aa], (E,B) <- (N,B)[aa]
    conn_area_e = phys.connList_areas[aa, :]                  # (E,1)
    km_a = km[aa, :]                                          # (E,B)
    km_b = km[bb, :]                                          # (E,B)
    k_ij = (km_a * km_b) / (phys.w_1 * km_b + phys.w_2 * km_a) * conn_area_e / (phys.conn_d1 + phys.conn_d2)

    # viscosity
    mu_a = (1.0 / mu_scalar) * k_ij
    mu_b = (1.0 / mu_scalar) * k_ij

    # direction
    select = (P[aa, :] > P[bb, :]).to(P.dtype)
    edge_attr = select * mu_a + (1.0 - select) * mu_b  # (E,B)

    # (B,N,N)
    adj = to_dense_adj(edge_index=phys.edge_index, edge_attr=edge_attr).squeeze(0).permute(2, 0, 1)
    adj_total = (adj + adj.transpose(1, 2)).float()

    # TT
    deg = torch.matmul(adj_total, phys.ones).squeeze(-1)  # (B, N)
    TT = -torch.diag_embed(deg) + adj_total  # (B, N, N)

    out1 = torch.matmul(TT, P.permute(1, 0).unsqueeze(2))  # (B, N, 1)
    out1 = out1.squeeze(2).permute(1, 0)  # (N, B)

    qw = torch.zeros((phys.N, P.shape[1]), dtype=P.dtype, device=P.device)
    qw[phys.prd_l_wellindex, :] = (phys.p_l - P[phys.prd_l_wellindex, :])
    qw[phys.prd_r_wellindex, :] = (phys.p_r - P[phys.prd_r_wellindex, :])

    return out1 + qw

