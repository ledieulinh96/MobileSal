import torch
import numpy as np

# adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
MAX_IMG_PER_BATCH = 256

# Determine the best available device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')  # For Apple Silicon Macs
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


class SalEval:
    def __init__(self, nthresh=255):
        self.nthresh = nthresh
        self.thresh = torch.linspace(1./(nthresh + 1), 1. - 1./(nthresh + 1), nthresh).to(device)
        self.EPSILON = np.finfo(float).eps

        self.gt_sum = torch.zeros((nthresh,)).to(device)
        self.pred_sum = torch.zeros((nthresh,)).to(device)
        self.num_images = 0
        self.mae = 0
        self.prec = torch.zeros(self.nthresh).to(device)
        self.recall = torch.zeros(self.nthresh).to(device)


    def addBatch(self, predict, gth):
        bs = predict.shape[0]
        assert(predict.shape[0] < MAX_IMG_PER_BATCH)
        predict = predict.detach()
        gth = gth.detach()
        gth.requires_grad = False
        predict.requires_grad = False
        recall = torch.zeros(self.nthresh).to(device)
        prec = torch.zeros(self.nthresh).to(device)

        mae = 0
        predict = predict.view(bs, -1)
        gth = gth.view(bs, -1)
        length = predict.shape[1]
        thres_mat = self.thresh.expand(bs, length, self.nthresh).permute(2, 0, 1)
        predict_ = predict.expand(self.nthresh, bs, length)
        gth_ = gth.expand(self.nthresh, bs, length)
        # nthres, n, length
        bi_res = (predict > thres_mat).float()
        intersect = (gth_ * bi_res).sum(dim=2) # nthres, n
        recall = (intersect / (gth_.sum(dim=2) + self.EPSILON)).sum(dim=1)
        prec = (intersect / (bi_res.sum(dim=2) + self.EPSILON)).sum(dim=1)
        mae = (predict_[0] - gth_[0]).abs().sum() / length

        self.prec += prec
        self.recall += recall
        self.mae += mae
        self.num_images += bs

    def getMetric(self):
        prec = self.prec / self.num_images
        recall = self.recall / self.num_images
        F_beta = (1 + 0.3) * prec * recall / (0.3 * prec + recall + self.EPSILON)
        MAE = self.mae / self.num_images
        return F_beta.max().item(),  MAE.item()

