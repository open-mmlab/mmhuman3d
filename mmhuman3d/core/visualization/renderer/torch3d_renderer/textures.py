import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import TexturesVertex


class TexturesClosest(TexturesVertex):
    """Textures for closest interpolation."""

    def sample_textures(self, fragments, faces_packed=None) -> torch.Tensor:
        """Rewrite sample_textures to use the closet interpolation.

        This function will only be called in render forwarding.
        """
        verts_features_packed = self.verts_features_packed()
        faces_verts_features = verts_features_packed[faces_packed]
        bary_coords = fragments.bary_coords
        _, idx = torch.max(bary_coords, -1)
        mask = torch.arange(bary_coords.size(-1)).reshape(1, 1, -1).to(
            self.device) == idx.unsqueeze(-1)
        bary_coords *= 0
        bary_coords[mask] = 1
        texels = interpolate_face_attributes(fragments.pix_to_face,
                                             bary_coords, faces_verts_features)
        return texels
