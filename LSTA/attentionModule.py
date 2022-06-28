from torch.autograd import Variable
#import torch_xla
#import torch_xla.core.xla_model as xm
from LSTA.ConvLSTACell import *
from LSTA import resNet
from tqdm import tqdm

class attentionModel(nn.Module):
    def __init__(self, num_classes=51, mem_size=512, c_cam_classes=1000, is_ta3n=False):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        #self.resNet = resNet.resnet34(True, True)
        self.mem_size = mem_size
        self.lsta_cell = ConvLSTACell(2048, mem_size, c_cam_classes)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        #todo: implementare ulteriori livelli della rete (Sudakaran email)

        # static params use from external methods
        self.is_ta3n            = is_ta3n
        self.dev                = torch.device("cuda:0") #xm.xla_device()
        self.loss_fn            = None
        self.optimizer_fn       = None
        self.optim_scheduler    = None

    def forward(self, features):
        # Features = Tensor (5, 32, 2048, 7, 7)
        state_att = (Variable(torch.zeros(features.size(1), 1, 7, 7).to(self.dev)),
                     Variable(torch.zeros(features.size(1), 1, 7, 7).to(self.dev)))
        state_inp = (Variable(torch.zeros((features.size(1), self.mem_size, 7, 7)).to(self.dev)),
                     Variable(torch.zeros((features.size(1), self.mem_size, 7, 7)).to(self.dev)))

        state_inp_stack = []
        for t in tqdm(range(features.size(0))):
            features_reshaped = features[t, :, :, :, :]
            state_att, state_inp, _ = self.lsta_cell(features_reshaped, state_att, state_inp)
            state_inp_stack.append(self.avgpool(state_inp[0]).view(state_inp[0].size(0), -1))

        feats = state_inp_stack[-1]
        logits = self.classifier(feats)
        if self.is_ta3n: # In order to mantains featurs as TA3N wants
            feats = torch.stack(state_inp_stack)
            feats = feats.permute(1, 0, 2)
        return logits, feats

    # General utils
    def set_loss_fn(self, loss):
        self.loss_fn = loss
    def set_optimizer_fn(self, optimizer):
        self.optimizer_fn = optimizer
    def set_optim_scheduler(self, optim_scheduler):
        self.optim_scheduler = optim_scheduler

# Noi partiamo direttamente da feature_conv riga 25 (dati da Chiara)
# Noi abbiamo (B, 5, 7, 7, 1024)... Come trattano la dimensione temporale? for t in ... ?