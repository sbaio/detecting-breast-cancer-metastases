#!/usr/bin/env python
# coding: utf-8

# load the data
import os
import numpy as np
import torch
from torch import nn
from glob import glob
import pandas as pd
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter
import sklearn
import sklearn.metrics
import sklearn.model_selection
import copy

data_dir = Path("data/")
train_tile_annotations = pd.read_csv(data_dir / "train_input/train_tile_annotations.csv")
now = datetime.now()
dt_string = now.strftime("%m%d_%H%M")
outdir = Path(f"runs/{dt_string}/")

def get_features(path, ntiles=1000):
    x = np.load(path)[:,3:]
    # tile features to have 1000 ones
    y = np.tile(x,(ntiles//x.shape[0],1))
    if y.shape[0] < ntiles:
        ncat = ntiles%x.shape[0]
        y = np.concatenate([y, x[:ncat]], axis=0)
    resnet_features = y # of size 1000 x 2048
    return resnet_features

class Dataset():
    def __init__(self):
        self.training_output = pd.read_csv(data_dir / "training_output.csv")
        self.ntiles = 1000

    def __getitem__(self, i):
        x = self.training_output.iloc[i]
        ID, target = x['ID'], x['Target']
        
        # load the pre-computed resnet features
        feat_path = glob(f"data/train_input/resnet_features/ID_{ID:03d}*.npy")[0]
        x = get_features(feat_path, ntiles=self.ntiles)
        resnet_features = torch.from_numpy(x).float()
        
        return resnet_features, target

    def __len__(self):
        return len(self.training_output)

class TestDataset():
    def __init__(self):
        self.test_features_paths = sorted(glob("data/test_input/resnet_features/ID_*.npy"))
        print(f"Test dataset has {len(self.test_features_paths)}")
        self.ntiles=  1000

    def __getitem__(self, i):
        feat_path = self.test_features_paths[i]
        ID = feat_path.split('/')[-1].split('.')[0].replace('ID_','')
        # load the pre-computed resnet features
        x = get_features(feat_path, ntiles=self.ntiles)
        resnet_features = torch.from_numpy(x).float()
        
        return resnet_features, ID

    def __len__(self):
        return len(self.test_features_paths)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        fSz = 2048 # dimension of resnet features
        self.conv1x1 = nn.Conv1d(fSz, 1, kernel_size=1, bias=False)

        self.R = 5
        self.mlp = nn.Sequential(
            nn.Linear(2*self.R, 200),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(200, 100),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 1),
        )
        # TODO: Replace Sigmoid with ReLU

        self.BCELossWithLogits = nn.BCEWithLogitsLoss()

    def forward(self, feats, targets=None):
        """
            feats: torch.FloatTensor features of the tiles of size bSz,ntiles, fSz 
        """
        
        # bSz, ntiles, fSz
        feats = feats.transpose(1,2)      # Adapt input for conv
        # bSz, fSz, ntiles
        feats = self.conv1x1(feats)[:,0]  # Feature embedding
        # bSz, ntiles

        # min-max selection
        vals, inds = feats.sort(dim=1)
        minmax_inds = torch.cat([inds[:,:self.R] , inds[:,-self.R:]], dim=1)
        minmax_feats = torch.gather(feats, dim=1, index=minmax_inds)

        logits = self.mlp(minmax_feats)
        # bSz, 1
        probs = torch.sigmoid(logits)

        out = {
            'logits':logits, 'probs':probs,
        }
        if targets is not None:
            # compute loss
            loss = self.BCELossWithLogits(logits, targets[:,None].float())
            # loss += 0.5 * self.conv1x1.weight.norm(2) # Add L2 regularization
            loss += 0.5 * self.conv1x1.weight.pow(2).sum() # Add L2 regularization
            out['loss'] = loss
            
        return out


# In[6]:


dset = Dataset()
model_ = Model()


# # TODO: data augmentation ?

def fit_and_score(train_dset, val_dset=None, run=0):
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=10, shuffle=True, drop_last=True)
    if val_dset is not None:
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size=10, shuffle=False, drop_last=False)

    model = copy.deepcopy(model_).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train loop
    nepochs = 30

    def validate():
        model.eval()

        all_preds = []
        all_targets = []
        for feats, targets in val_loader:
            all_targets.append(targets.numpy())

            with torch.no_grad():
                out = model(feats.cuda())
            all_preds.append(out['probs'].cpu().numpy())
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        auc = sklearn.metrics.roc_auc_score(all_targets, all_preds)

        return auc

    writer = SummaryWriter(outdir / f'run_{run}')

    iteration = 0
    best_auc = 0
    for epoch in range(nepochs):
        model.train()
        for feats, targets in train_loader:
            feats = feats.cuda()
            targets = targets.cuda()
            out = model(feats, targets=targets)

            optimizer.zero_grad()
            out['loss'].backward()
            optimizer.step()

            writer.add_scalar("Train/loss", out['loss'], iteration)
            iteration += 1

        if val_dset is None:
            continue

        # validate and keep the best at each epoch
        auc = validate()
        writer.add_scalar("Val/AUC", auc, iteration)
        
        if auc > best_auc:
            best_auc = auc
            # save best model
            ckpt = {
                'model':model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch':epoch,
                'auc':auc,
            }
            torch.save(ckpt, os.path.join(writer.logdir, 'model_best.pth'))
        # print(f"Epoch {epoch}: AUC {auc:0.2f}, best AUC {best_auc:0.2f}")

    if val_dset is None:
        ckpt = {
            'model':model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch':epoch,
        }
        torch.save(ckpt, os.path.join(writer.logdir, 'model_final.pth'))

    return best_auc


# In[3]:



num_runs = 10
num_splits = 5

aucs = []
for seed in range(num_runs):
    # create new model
    cv = sklearn.model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True,random_state=seed)

    targets = dset.training_output['Target'].tolist()

    cv_aucs = []
    for i, (train_inds, val_inds) in enumerate(cv.split(dset, y=targets)):
        train_dset = torch.utils.data.Subset(dset, train_inds)
        val_dset = torch.utils.data.Subset(dset, val_inds)

        # train model on train_dset; evaluate on val_dset
        auc = fit_and_score(train_dset, val_dset, run=f'seed_{seed}_cv_{i}')
        print(f"seed: {seed} - CV [{i+1}/{num_splits}] -  AUC: {auc:0.2f}")
        cv_aucs.append(auc)
    
    aucs.append(cv_aucs)

aucs = np.array(aucs)

print("Predicting weak labels using Chowder")
print("AUC: mean {}, std {}".format(aucs.mean(), aucs.std()))


# In[22]:


# generate the submission file

# -------------------------------------------------------------------------
# Prediction on the test set

# Train a final model on the full training set
train_dset = dset
label = 'final'
print("Training final model")
_ = fit_and_score(train_dset, val_dset=None, run=label)
model = Model()
ckpt = torch.load(outdir / f'run_{label}/model_final.pth')
model.load_state_dict(ckpt['model'])
model = model.cuda().eval()

test_dset = TestDataset()

loader = torch.utils.data.DataLoader(test_dset, batch_size=len(test_dset), shuffle=False)

feats, IDs = iter(loader).next()

# load test features
with torch.no_grad():
    preds_test = model(feats.cuda())['probs'].cpu().numpy()[:,0]

print(preds_test.shape)

# Check that predictions are in [0, 1]
assert np.max(preds_test) <= 1.0
assert np.min(preds_test) >= 0.0

# -------------------------------------------------------------------------
# Write the predictions in a csv file, to export them in the suitable format
# to the data challenge platform
test_output = pd.DataFrame({"ID": IDs, "Target": preds_test})
test_output.set_index("ID", inplace=True)
test_output.to_csv(data_dir / "preds_test_chowder.csv")



# use the annotated tiles from data/train_input/train_tile_annotation.csv to compare tumoral and non tumoral tiles.

