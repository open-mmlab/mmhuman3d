import torch


def vis_z_buffer(z, percentile=1, vis_pad=0.2):
    z = z[:, :, 0]
    mask = z > 1e-5
    if torch.sum(mask) == 0:
        z[...] = 0
    else:
        vmin = torch.quantile(z[mask], percentile / 100)
        vmax = torch.quantile(z[mask], 1 - percentile / 100)
        pad = (vmax - vmin) * vis_pad
        vmin_padded = vmin - pad
        vmax_padded = vmax + pad
        z[mask] = vmin + vmax - z[mask]
        z = (z - vmin_padded) / (vmax_padded - vmin_padded)
        z = torch.clip(torch.round(z * 255), 0, 255)
    z_cpu = z.to(dtype=torch.uint8).detach().cpu().numpy()
    return z_cpu


def vis_normals(coords, normals, vis_pad=0.2):
    mask = coords[:, :, 2] > 0
    coords_masked = -coords[mask]
    normals_masked = normals[mask]

    coords_len = torch.sqrt(torch.sum(coords_masked**2, dim=1))

    dot = torch.sum(coords_masked * normals_masked, dim=1) / coords_len

    h, w = normals.shape[:2]
    vis = torch.zeros((h, w), dtype=coords.dtype, device=coords.device)
    vis[mask] = torch.clamp(dot, 0, 1) * (1 - 2 * vis_pad) + vis_pad

    vis = (vis * 255).to(dtype=torch.uint8)

    return vis
