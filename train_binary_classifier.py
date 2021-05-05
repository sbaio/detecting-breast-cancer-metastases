# build a binary classifier on the tiles
import os
import pandas as pd
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
from torchvision.models import resnet18

from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from tqdm import tqdm

def get_ID_file_from_name(name):
    ID = name[:6]
    file = f"data/train_input/images/{ID}_annotated/{name}"
    return ID, file

class DiscreteRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class TileDataset():
    def __init__(self, df, transform=None):
        df = df.sort_values(by=['Target'])
        self.df = df
        self.transform = transform

    def __getitem__(self, i):
        x = self.df.iloc[i]
        name = x.name
        target = x['Target']
        ID, file = get_ID_file_from_name(name)
        im = Image.open(file)
        if self.transform is not None:
            im = self.transform(im)
        return im, target, i

    def __len__(self):
        return len(self.df)


anno = pd.read_csv("data/train_input/train_tile_annotations.csv", index_col=0)

seed = 0
num_splits = 10
outdir = "runs/train_binary_classifier/test/"

cv = StratifiedKFold(n_splits=num_splits, shuffle=True,random_state=seed)
for train_inds, val_inds in cv.split(anno, y=anno['Target']):
    batch_size = 64
    nepochs = 100

    # transforms
    normalize = T.Normalize(mean=[0.5972, 0.4646, 0.5658], std=[0.2730, 0.2970, 0.2617])
    train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            DiscreteRotation([0, 90, -90, 180]),
            T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0),
            T.ToTensor(),
            normalize,
        ])
    val_transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])

    train_dset = TileDataset(anno.iloc[train_inds], transform=train_transform)
    val_dset = TileDataset(anno.iloc[val_inds], transform=val_transform)
    print(len(train_dset), len(val_dset))
    

    df = train_dset.df['Target']
    counts = df.value_counts()
    # weight the rare samples so that they are sampled with equal freq as pos samples
    weights = (train_dset.df['Target']*(counts[0]/counts[1]-1)+1)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size*int(len(train_dset)/batch_size))
    train_loader = torch.utils.data.DataLoader(train_dset, 
                batch_size=batch_size, sampler=sampler, num_workers=8)
    val_loader = torch.utils.data.DataLoader(train_dset, 
                batch_size=batch_size, shuffle=False, num_workers=8)

    # network
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, 1)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
    def validate():
        all_preds = []
        all_targets= []

        model.eval()
        for i, (ims, targets, _) in enumerate(val_loader):
            ims = ims.cuda()
            
            with torch.no_grad():
                preds = model(ims).cpu()
            all_preds.append(preds)
            all_targets.append(targets)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
        return auc

    writer = SummaryWriter(outdir)
    best_auc = 0
    iteration = 0
    for epoch in tqdm(range(nepochs)):
        model.train()
        for i, (ims, targets, _) in enumerate(tqdm(train_loader)):
            ims = ims.cuda()
            targets = targets.cuda()
            preds = model(ims)

            loss = nn.functional.binary_cross_entropy_with_logits(preds[:,0], targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                writer.add_scalar("Train/loss", loss.item(), iteration)
            iteration += 1

        auc = validate()
        
        writer.add_scalar("Val/auc", auc, epoch)
        writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()
        if auc > best_auc:
            best_auc = auc

            ckpt = {
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'epoch':epoch,
                'iteration':iteration,
                'auc':auc,
            }
            torch.save(ckpt, os.path.join(writer.logdir, 'model_best.pth'))

            
    break
