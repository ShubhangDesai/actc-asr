from torch.utils.data import DataLoader

from .RawDataset import *

def get_loader(args, set_name):
    dataset = RawDataset(args['datadir'], set_name.replace('_random', ''), args['uniform'], args['delta'], args['features'] == 'pw')
    loader =  DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0) # (set_name=='train' or 'random' in set_name)

    return loader
