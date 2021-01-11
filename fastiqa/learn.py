"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from fastiqa.models.all import *
from fastiqa.bunches.im_roi2mos import *
from fastiqa.bunches.base import to_json
import torchvision.models as models
from fastiqa.learn import *
dls = ImRoI2MOS.from_json('json/LIVE_FB_IQA.json', bs=3)
model = BodyHeadModel(backbone=models.resnet18)
learn = IqaLearner(dls, model)
to_json(learn, 'json/example_learner.json')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from .resnet_3d import *
from .resnet3d.resnext import *
from .ts import *
"""

# from .baiscs import to_json
from fastai.vision.all import *
from .bunch import IqaDataBunch
import scipy.stats as scs
# from dataclasses import dataclass
import seaborn as sns
import logging

# only compute based on the first axis (video mos, don't include patch)
@delegates(AccumMetric)
def SRCC(dim_argmax=None, axis=0, nan_policy='propagate', **kwargs):
    "Spearman correlation coefficient for regression problem"
    def spearmanr(x,y=None,**kwargs):
        if x.dim() == 2: x = x[:, 0]
        if y.dim() == 2: y = y[:, 0]
        return scs.spearmanr(x, y,**kwargs)[0]
    return AccumMetric(partial(spearmanr, axis=axis, nan_policy=nan_policy),
                       invert_arg=False, dim_argmax=dim_argmax, flatten=False, **kwargs)

@delegates(AccumMetric)
def LCC(dim_argmax=None, **kwargs):
    "Pearson correlation coefficient for regression problem"
    def pearsonr(x,y):
        if x.dim() == 2: x = x[:, 0]
        if y.dim() == 2: y = y[:, 0]
        return scs.pearsonr(x, y)[0]
    return AccumMetric(pearsonr, invert_arg=False, dim_argmax=dim_argmax, flatten=False, **kwargs)

def DummyLoss(x, y, *args, **kwargs):
    return torch.tensor(np.nan)

def DebugLoss(x, y, *args, **kwargs):
    print(x.shape, y.shape)
    print(x, y)
    return torch.tensor(np.nan)

#
class IqaLearner(Learner):

    def to_json(self):
        return {'dls':self.dls, 'model':self.model}

    def __init__(self, dls, model, *args, metrics=[SRCC(), LCC()],
            cbs=None, loss_func=MSELossFlat(),
            **kwargs):
        # create a new instance each time!!!
        # otherwise it will use the same object

        if hasattr(model, 'bunch'):
            if isinstance(dls, IqaDataBunch):
                dls._bunching = True
            # dls being a dict or string?
            # no __name__ attribute
            # add __str__ ?
            # TODO: show changed properties when bunching
            logging.info(f'Bunching ... {model.__name__}')
            dls = model.bunch(dls)
            logging.info(f'Bunched ... {model.__name__}')
            if isinstance(dls, IqaDataBunch):
                dls._bunching = False
        else:
            logging.warning(f'model {model.__name__} did not implement bunch(dls) method')

        if cbs is None:
            # the first metric
            monitor = metrics[0].name if len(metrics) > 1 else 'valid_loss' #'spearmanr' else loss
            cbs = [CSVLogger(append=True), ShowGraphCallback(), SaveModelCallback(monitor=monitor)]
            print(f'monitoring {monitor}')
        super().__init__(dls, model, *args, metrics=metrics, cbs=cbs, loss_func=loss_func, **kwargs)

        # [Discriminative layer training](https://docs.fast.ai/basic_train.html#Discriminative-layer-training)
        # if hasattr(self.model, 'splitter'):
        #     #self.split(self.model.split_on)
        #     self.splitter=model.splitter

        # ShowGraphCallback()
        # SaveModelCallback()

        # self.callback_fns += [ShowGraph, partial(CSVLogger, append=True),
        #                       partial(SaveModelCallback, every='improvement',
        #                               monitor=self.metrics[0].name)]

        # predict on database (cached) and get numpy predictions

    def extract_features(self, on=None, name='features', cache=True, skip_exist=False):
        # Learner.get_preds will get preds on valid set without shuffling and drop_last_batch
        # clip/frame features combined into one feature
        # output numpy features (192, 512, 2, 2)
        if on is None: on = self.dls
        npy_file = self.dls.path / (f'features/{name}') / (on.__name__ + '.npy')
        # there might be / in on.__name__ to help create sub folders
        npy_file.parent.mkdir(parents=True, exist_ok=True)
        # self.path / (f'{name}@' + on.__name__ + '.npy')
        if cache and npy_file.absolute().exists():
            if skip_exist: return None
            with open(npy_file, 'rb') as f:
                features = np.load(f)
        else:
            # try:
            current_data = self.dls
            self.dls = on
            old_setting = self.model.output_features
            self.model.output_features = True # TODO:  put it outside
            # reset roi
            if hasattr(self.model, 'rois'):
                self.model.rois = None
            preds = self.get_preds() # get preds on one video with several batches
            self.model.output_features = old_setting

            features = preds[0]
            self.dls = current_data

            if cache:
                with open(npy_file, 'wb') as f:
                    np.save(f, features)

            # except: # memory issue
            #     return None
        return features

    def get_np_preds(self, on=None, cache=True, jointplot=False, **kwargs):
        """
        get numpy predictions on a bunched database
        TODO: check bunched
        IqaLearner assumes that the output is only a scalar number
            output = preds[0]
            target = preds[1]
        """
        if on is None:
            on = self.dls

        # if we don't flatten it, then we cannot store it.
        # so we need to only get the valid output
        # rois_learner gives three output csv

        # load dls here...... TOFIX

        on = self.model.bunch(on)
        metric_idx = on.metric_idx
        # suffixes = ['', '_patch_1', '_patch_2', '_patch_3']
        suffixes = ['', '_p1', '_p2', '_p3']

        csv_file = self.path / ('valid@' + on.__name__ + suffixes[metric_idx] + '.csv')
        if os.path.isfile(csv_file) and cache:
            print(f'load cache {csv_file}')
            df = pd.read_csv(csv_file)
            output = np.array(df['output'].tolist())
            target = np.array(df['target'].tolist())
        else:
            c = on.c if type(on.c) == int else on.c[-1]
            logging.debug(f'validating... {self.model.__name__}@{on.__name__} (c={c})')
            # on.c = 1
            # TODO fuse duplicate code with rois_learner
            current_data = self.dls
            self.dls = on
            preds = self.get_preds()
            self.dls = current_data

            output, target = preds

            if not isinstance(output,(np.ndarray)):
                output = output.flatten().numpy()
                target = target.flatten().numpy()
            """
            preds is a list [output_tensor, target_tensor]
            torch.Size([8073, 4])
            """
            # don't call self.data.c to avoid unnecessary data loading
            # n_output = self.data.c  # only consider image score
            # no need since metric will take care of it
            # print(np.array(output).shape, np.array(target).shape) # (233, 1) (233,)
            if cache:
                # # we already loaded the data, so feel free to call data.c?
                if c == 4:
                    print('on.c==4')
                    for n in [0, 1, 2, 3]:
                        df = pd.DataFrame({'output': output[n::c], 'target': target[n::c]})
                        csv_file = self.path / ('valid@' + on.__name__ + suffixes[n] + '.csv')
                        df.to_csv(csv_file, index=False)
                elif c == 2:
                    print('on.c==2')
                    for n, roi_index in enumerate(on.feats[0].roi_index):
                        df = pd.DataFrame({'output': output[n::c], 'target': target[n::c]})
                        csv_file = self.path / ('valid@' + on.__name__ + suffixes[roi_index] + '.csv')
                        df.to_csv(csv_file, index=False)
                elif c == 1:
                    print('on.c==1')
                    df = pd.DataFrame({'output': output, 'target': target})
                    df.to_csv(csv_file, index=False)
                else:
                    raise NotImplementedError
            output = output[on.metric_idx::c]
            target = target[on.metric_idx::c]

        if cache and jointplot:
            # p = sns.jointplot(x="output", y="target", data=df)
            #plt.subplots_adjust(top=0.9)

            # size: 30k 2   1k 5
            g = sns.jointplot(x="output", y="target", data=df, kind="reg", marker = '.', scatter_kws={"s": 5},
                  xlim=(0, 100), ylim=(0, 100)) # color="r",
            plt.suptitle(f"{self.model.__name__}@{on.__name__}")
            #g.fig.suptitle(f"{self.model.__name__}@{on.__name__}") # https://stackoverflow.com/questions/60358228/how-to-set-title-on-seaborn-jointplot
            #g.annotate(stats.pearsonr)
            # g = sns.JointGrid(x="output", y="target", data=df) # ratio=100
            # g.plot_joint(sns.regplot)
            # g.annotate(stats.pearsonr)
            # g.ax_marg_x.set_axis_off()
            # g.ax_marg_y.set_axis_off()
        return output, target


    def valid(self, on=None, metrics=None, cache=True, all_items=False, **kwargs):
        """
        all_items/ True: test on all items, False: test on items in valid subset.
        """
        def valid_one(data):
            # logging.debug(f'validating... {self.model.__name__}@{data.__name__}')
            output, target = self.get_np_preds(on=data, cache=cache, **kwargs)  # TODO note here only output 1 scores
            output, target = torch.from_numpy(output), torch.from_numpy(target)
            return {metric.name: metric(output, target) for metric in metrics}

        if metrics is None: metrics = self.metrics

        # avoid changing self.data
        if on is None:
            on = self.dls

        if not isinstance(on,  (list, tuple)  ):
            on = [on]

        # call model.bunch after dls.bunch
        # if don't bunch, name is not available
        # bunch won't take time !!!! just update attributes, change svr model to allow that. only cache X when needed !!!

        on = [self.dls.bunch(x, **kwargs) if isinstance(x, (str,dict)) else x for x in on]
        #don't change c
        # must call model.bunch
        # on = [self.model.bunch(x, **kwargs) for x in on]
        records = [valid_one(data) for data in on]
        return pd.DataFrame(records, index=[data.__name__ for data in on]) # abbr






 # TestLearner(dls, model, metrics=[SRCC(), LCC()])
class TestLearner(IqaLearner):
    # don't use None, it will be filled in by IqaLearner
    # assert self.monitor in self.recorder.metric_names[1:]
    def __init__(self, *args, metrics=(), cbs=(), loss_func=DummyLoss,
            **kwargs):
        super().__init__(*args, metrics=metrics, cbs=cbs, loss_func=loss_func, **kwargs)
