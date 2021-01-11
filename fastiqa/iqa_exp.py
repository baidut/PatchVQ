"""
# %%

# TODO # convert to IqaLearner without changing information?

import time
from tqdm.notebook import tqdm

for i in tqdm(range(10)):
    time.sleep(1)
    pass

# %%
import time
from tqdm.notebook import tqdm

for i in tqdm(range(10)):
    time.sleep(1)
    pass

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.vqa import *
# without > with
IqaExp.from_dir('exp/InceptionTimeModel/train@LIVE_FB_VQA_30k/2p3d')
IqaExp.from_dir('exp/InceptionTimeModel/train@LIVE_FB_VQA_30k/2+3d')
IqaExp.from_dir('exp/InceptionTimeModel/train@LIVE_FB_VQA_30k/2d')
# with > without
IqaExp.from_dir('exp/InceptionTimeModel/train@LIVE_FB_VQA_v1/2+3d')
################################################################################


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%
"""

# from fastai.callbacks import *

from pathlib import Path
import random
import torch  # pytorch RNGs
import numpy as np  # numpy RNG
import fastai
import logging
from .basics import *
from fastai.test_utils import * # synth_learner

# learner
def read_log(self):
    # TODO AttributeError: 'Learner' object has no attribute 'csv_logger'
    assert hasattr(self, 'csv_logger')
    if self.csv_logger is None: return None
    _df = self.csv_logger.read_log()
    df = _df.dropna() # read_logged_file() fastai1
    # a = [literal_eval(x) for x in learner.records['train loss'].tolist()]
    # train_losses = np.array(a).flatten()
    # valid_losses = learner.records['valid loss'].to_list()
    df = df[df.time.str.contains(':')] # df[df.epoch.astype(str) != 'epoch']
    if len(df) != len(_df):
        df.to_csv(self.path/'history.csv', index=False) # load it again so that the data type is correct
        df = self.csv_logger.read_log()
    return df

class IqaExp(dict):
    path = '!exp'
    seed = None
    keep_learn_path = False # don't change learn.path

    def __init__(self, path=None, gpu=None, seed=None, keep_learn_path=False):
        super().__init__()
        self.path = Path(path) if path else Path(self.path)

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if type(gpu) is int:
            assert gpu < n_gpu
            torch.cuda.set_device(gpu)

        self.gpu = gpu
        self.seed = seed
        self.set_seed()
        self.keep_learn_path = keep_learn_path

        # self.testdb = testdb
        # self.metrics = metrics

    def set_seed(self, dls=None):
        if self.seed is not None:
            set_seed(self.seed, True)
            if dls is not None:
                dls.rng.seed(self.seed) # will force to load the database

    def __iadd__(self, other):
        self[other.model.__name__] = other
        return self

    def __setitem__(self, key, learn):
        """ if learn.path is not set, set it"""
        # fastai v1 learn.data -- > v2 learn.dls
        if not self.keep_learn_path: # str(learn.path) in ['.' or str(learn.dls.path)]:
            if hasattr(learn.dls, '__name__'):
                data_name = learn.dls.__name__
            else:
                try:
                    if type(learn.dls) is fastai.data.core.DataLoaders:
                        data_name = str(learn.dls.path).split('/')[-2]
                    else:
                        data_name = str(learn.dls.path).split('/')[-1]
                except:
                    raise NotImplementedError("please set dls.__name__")

            config = 'default' if learn.model.__name__ == key else key
            exp_path = self.path / learn.model.__name__ / ('train@' + data_name) / config  # exp/resnet18/CLIVE
            learn.path = exp_path
        super().__setitem__(key, learn)

    @property
    def one_item(self):
        return next(iter(self.values()))


    # TODO self[key] = self[key].to_fp16()
    # func=fine_tune
    def run(self, func, destroy=False, append=False):
        # TODO option load best or load latest
        for key, learn in self.items():
            exp_path = learn.path  # exp/resnet18/CLIVE
            if append is False:
                # check if history exists
                if (Path(exp_path)/'models/model.sav').exists(): # SVRLearner
                    print(f'[SVRLearner] Skip cuz FileExists {key} {exp_path}')
                    continue
                elif (Path(exp_path)/'models/model.pth').exists(): # and Path(exp_path)/'history.csv').exists()
                    print(f'Skip cuz FileExists {key} {exp_path}')
                    continue
                else:
                    Path(exp_path).mkdir(parents=True, exist_ok=True)

                # try:
                #     print(f'Creating: {key} {exp_path}')
                #     os.makedirs(exp_path)
                # except FileExistsError:
                #     print(f'Skip cuz FileExists {key} {exp_path}')
                #     continue
                # except:
                #     raise
            if not (hasattr(learn.model, 'output_features') and learn.model.output_features):
                to_json(learn, learn.path/'config.json')

            self.set_seed(learn.dls)

            if type(self.gpu) is int:
                func(learn)
            else: # None or True/False (cpu or multi-gpus)
                ctx = learn.parallel_ctx
                with partial(ctx, self.gpu)():
                    func(learn)

            # learn.save(key)
            if destroy:
                learn.destroy()

            self[key] = learn  # call __setitem__
        return self

    def load(self, name=None):
        if name is None: name = 'model' # name = 'bestmodel'
        return self.run(lambda x: x.load(name), append=True)

    def fit(self, n=15, append=False, **kwargs):
        return self.run(lambda x: x.fit(n, **kwargs), append=append)

    def fit_one_cycle(self, n, append=False, **kwargs):
        return self.run(lambda x: x.fit_one_cycle(n, **kwargs), append=append)

    def _repr_html_(self, clear=True):
        keys = []
        print(self.path)
        for key, learn in self.items():
            if not hasattr(learn, 'csv_logger'):
                learn.csv_logger = CSVLogger(append=True)  # CSVLogger(learn, append=True) fastai1
            df = read_log(learn)
            if df is not None: keys.append(key)

        print(keys)
        if keys != []:
            logging.info(keys)
            self.show_losses(keys)
            self.show_metrics(keys)
            self.show_current_best(keys)
        # try:
        # except:
        #     pass
        return ''

    @classmethod
    def from_dir(cls, path):
        path = Path(path)
        e = cls(keep_learn_path=True)
        for f in path.glob('*/'):
            if f.is_dir():
                cfg = f.stem
                learn = synth_learner(cbs=[CSVLogger(append=True)],  metrics=[SpearmanCorrCoef(), PearsonCorrCoef()], path = path/cfg)
                e[cfg.split('--')[0]] = learn
        return e


    def __getattr__(self, k: str):  # total_params
        # %% when returning a single value
        # d = {key: getattr(learn, k) for key, learn in self.items()}
        # return pd.DataFrame([d], index=[k]).T

        # %% when return a dict
        return pd.DataFrame([getattr(learn, k)
                             for learn in self.values()], self.keys())

    def valid(self, on=None, metrics=None, cache=True, **kwargs):

        def valid_one(l):
            df = l.valid(on, metrics=metrics, cache=cache, **kwargs).T
            return df[df.columns].apply(lambda row: ','.join(row.map('{:.3f}'.format)))

        if on is None: # simply run each validation
            self.run(lambda x: x.valid(), **kwargs)
        # %%
        # frames = [learn.valid(data).add_prefix(key+'_') for key, learn in self.items()]
        # return pd.concat(frames)
        d = [valid_one(learn) for learn in self.values()]
        # pd.options.display.float_format = '{:,.3f}'.format
        return pd.DataFrame(d, index=list(self.keys()))

    def show_current_best(self, keys=None):
        # show current best performance

        if keys is None: keys = self.keys()
        rows = []
        for key in keys:
            learn = self[key]
            df = read_log(learn)
            idx = df['spearmanr'].astype(float).idxmax()
            rows.append(df.loc[idx].to_dict())

        df = pd.DataFrame(rows, index=keys)
        display(df)
        if 'spearmanr' in df.columns and 'pearsonr' in df.columns:
            display(df[['spearmanr', 'pearsonr']])
        return

    def show_losses(self, keys=None):
        # Train error and Test error
        # from ast import literal_eval
        if keys is None: keys = self.keys()
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
        # fig.set_size_inches(18.5, 10.5, forward=True)
        ax1.set_title("Train error")
        ax2.set_title("Test error")
        for key in keys:
            learn = self[key]
            df = read_log(learn)
            train_losses = df['train_loss'].astype(float).to_list()
            valid_losses = df['valid_loss'].astype(float).to_list()
            ax1.plot(np.log10(train_losses), label=key)
            ax2.plot(np.log10(valid_losses), label=key)

        ax1.set_ylabel('Log Loss')
        ax1.set_xlabel('# Epochs') # Batches processed


        ax2.set_ylabel('Log Loss')
        ax2.set_xlabel('# Epochs') # Batches processed
        ax2.legend(loc="upper left", bbox_to_anchor=(1,1))

    def show_metrics(self, keys=None, sharey=True):  # metrics
        if keys is None: keys = self.keys()
        metrics = self.one_item.metrics
        if not metrics:
            return  # no metrics
        if sharey:
            fig, axes = plt.subplots(1, len(metrics), sharey=True, figsize=(8, 3))
        else:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 5 * len(metrics)))

        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            name = metric.name
            axes[idx].set_title(name)
            axes[idx].set_ylabel('Score')
            axes[idx].set_xlabel('# Epochs') # Batches processed
            for key in keys:
                learner = self[key]
                # score = learner.records[name].to_list()
                df = read_log(learner)
                score = df[name].astype(float).to_list()
                # print(score)
                axes[idx].plot(score, label=key)

        #xes[0].legend(loc="upper right", bbox_to_anchor=(1,1))
