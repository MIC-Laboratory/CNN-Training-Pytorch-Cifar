from torch.optim.lr_scheduler import CosineAnnealingLR
class my_CosineAnnealingLR(CosineAnnealingLR):
    def __init__(self,optimizer):
        super().__init__(optimizer,T_max=200)