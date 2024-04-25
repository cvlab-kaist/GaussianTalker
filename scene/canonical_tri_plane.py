import torch.nn as nn
from scene.hexplane import HexPlaneField


class canonical_tri_plane(nn.Module):
    def __init__(self, args=None,D = 8,W=64):
        super(canonical_tri_plane, self).__init__()
        self.W = W
        self.D = D
        self.args = args
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        
        self.feature_out = [nn.Linear(args.d_model,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.scales = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def mlp_init_zeros(self):
        
        nn.init.xavier_uniform_(self.scales[-1].weight,gain=0.1)
        nn.init.zeros_(self.scales[-1].bias)
        
        nn.init.xavier_uniform_(self.rotations[-1].weight,gain=0.1)
        nn.init.zeros_(self.rotations[-1].bias)
        
        nn.init.xavier_uniform_(self.opacity[-1].weight,gain=0.1)
        nn.init.zeros_(self.opacity[-1].bias)
        
        nn.init.xavier_uniform_(self.shs[-1].weight,gain=0.1)
        nn.init.zeros_(self.shs[-1].bias)
        
    def mlp2cpu(self):
        self.feature_out = self.feature_out.to('cpu')
        self.scales = self.scales.to('cpu')
        self.rotations = self.rotations.to('cpu')
        self.opacity = self.opacity.to('cpu')
        self.shs = self.shs.to('cpu')
        
    def forward(self, rays_pts_emb, only_feature = False, train_tri_plane=True):
        scale, rotation, opacity, sh = None,None,None,None
        B,_,_ = rays_pts_emb.shape
        feature = self.grid(rays_pts_emb[0,:,:3]).unsqueeze(dim=0).repeat(B,1,1)
        if only_feature:
            if train_tri_plane == False:
                feature = feature.detach()
            return feature
        
        feature = self.feature_out(feature)
        scale = self.scales(feature)
        rotation = self.rotations(feature)
        opacity = self.opacity(feature)
        sh = self.shs(feature)
        
        return feature, scale, rotation, opacity, sh
        


    