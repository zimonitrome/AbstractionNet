from torchvision import models
from torch import nn

class VGGContentLossMultiLayer(nn.Module):
    def __init__(self, layers = [26]):
        super(VGGContentLossMultiLayer, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        
        for l in vgg:
            if isinstance(l, nn.ReLU):
                l = nn.ReLU(inplace=False)

        blocks = []
        last_ix = 0
        for ix in layers:
            blocks.append(vgg[last_ix:ix].eval())
            last_ix = ix
        for bl in blocks:
            for p in bl:
                p.requires_grad = False

        self.blocks = nn.ModuleList(blocks)
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        loss = 0.0
        for block in self.blocks:
            x = block(x.clone())
            y = block(y.clone())
            loss += self.criterion(x, y)
        return loss