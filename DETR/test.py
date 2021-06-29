import torch
from torchsummary import summary
from typing import Optional, List
from torch import Tensor



class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device)

    # one method
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    # seconde method
    tensor = torch.randint((1,3,800,1200))
    mask = torch.randint((1,1,800,1200)).to(torch.bool)
    x = NestedTensor(tensor,mask).to(device=device)
    summary(model,x)
