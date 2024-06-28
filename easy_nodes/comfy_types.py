import torch

from comfy.clip_vision import ClipVisionModel
from comfy.sd import VAE
from easy_nodes.easy_nodes import AnythingVerifier, TensorVerifier, TypeVerifier, register_type, AnyType, any_type


class ImageTensor(torch.Tensor): pass
class MaskTensor(torch.Tensor): pass

class LatentTensor(torch.Tensor): pass
class ConditioningTensor(torch.Tensor): pass
class ModelTensor(torch.Tensor): pass
class SigmasTensor(torch.Tensor): pass

# Maybe there's an actual class for this?
class PhotoMaker: pass

# Abstract type, not for instantiating.
class NumberType: pass


# ComfyUI will get the special string that anytype is registered with, which is hardcoded to match anything.
register_type(AnyType, any_type, verifier=AnythingVerifier())

# Primitive types
register_type(int, "INT")
register_type(float, "FLOAT")
register_type(str, "STRING")
register_type(bool, "BOOLEAN")
register_type(NumberType, "NUMBER", verifier=TypeVerifier([float, int]))

register_type(ImageTensor, "IMAGE", verifier=TensorVerifier("IMAGE", allowed_shapes=[4], allowed_channels=[1, 3, 4]))
register_type(MaskTensor, "MASK", verifier=TensorVerifier("MASK", allowed_shapes=[3], allowed_range=[0, 1]))
register_type(LatentTensor, "LATENT", verifier=TensorVerifier("LATENT"))
register_type(ConditioningTensor, "CONDITIONING", verifier=TensorVerifier("CONDITIONING"))
register_type(ModelTensor, "MODEL", verifier=TensorVerifier("MODEL"))
register_type(SigmasTensor, "SIGMAS", verifier=TensorVerifier("SIGMAS"))

register_type(ClipVisionModel, "CLIP_VISION", verifier=AnythingVerifier())

# Did I get the right class for VAE?
register_type(VAE, "VAE", verifier=AnythingVerifier())
register_type(PhotoMaker, "PHOTOMAKER", verifier=AnythingVerifier())
