# High-level semantic feature matters few-shot unsupervised domain adaptation
Our paper "High-level semantic feature matters few-shot unsupervised domain adaptation" is accepted by AAAI2023. The arxiv version can be downloaded from the website "[https://arxiv.org/abs/2301.01956](https://arxiv.org/abs/2301.01956)". 

# Few-shot framework for training, tuning and report

## How to run a experiment?
* Run command below to train a FSL/FSUDA model.
```shell
# see 'run_script_experimetal.py in the root path'
python run_script_experimetal.py --gpu=GIDS --metric=BOW2ComponentIMSE
```
* Check results/saved model weights in "./adjust_parameters"

* If you want to add some new methods, please set it as below

```python
import model.hyper_model
model.hyper_model.Trainer.method.imgtoclass = YOURMETHOD
```

## How to convert my torch module to experiment for inserting experiment meters and checkout results?
* For example, a torch.nn.Module can be declare as:
```python
import torch
class MyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ....
    
    def forward(self, *args, **kwargs):
        ...
```
* Simply, you can change the base class to set your model as experiment one:
```python
from model.BaseModel import BaseModelFSL
# change the father class name
class MyModel(BaseModelFSL):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ....
    
    # here's different
    def forward_(self, *args, **kwargs):
        ...
```
* And you can plug the scalar tensor as below to collect the experiment meters as you wish:
```python
# all of meters you set will be collect automatcally by experiment manager.
from model.BaseModel import BaseModelFSL
class MyModel(BaseModelFSL):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ....
    
    # 
    def forward_(self, x, *args, **kwargs):
        # It'ld be a scalar tensor
        self.set_meter('x_mean', x.mean())
        ...
    
    def sub_model1(self, input_):
        logits = input_.softmax(-1)
        self.set_meter('logits_mean', logits.mean())
```
* Then you can check the meter change as epoch by keywords such as 'x_mean' and 'logits_mean'

## How to tune hyper-parameter automatcally?
* In the ```run_script_experimetal.py```
```python
from experiment_manage import ExperimentServer
from main import main
import argparse, os
import make_args

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1')
parser.add_argument('--metric', default='MELDA')
p_args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = p_args.gpu

# Hyperparameters to be tuned, see 'https://nni.readthedocs.io/zh/stable/hpo/search_space.html'
parameters_dir = {
    'metric': [p_args.metric], 'data_name': ['visda'], 
   ...
}

para_spaces = {
    'TO_BE_TUNED' : {
        '_type': 'uniform',
        '_value': [0.5, 0.9]
    },
    ...
}

# parameters_dir: Fixed experiment hyper-parameters
# func_tobe_excuted: Main function to be called
# auto_search: Decide if use the nni to search optimal hyper-parameters
# paras_space: To be tuned hyper-parameters
exp_server = ExperimentServer(parameters_dir=parameters_dir, func_tobe_excuted=main, auto_search=True, paras_space=para_spaces)
exp_server.opt.gpu = p_args.gpu
exp_server.run()
```
## How to set nni configuration for experiment?
* Modify code in the `experiment_manage.py`, see **run_nni_experiment** method of the class **ExperimentServer**
```python
    def run_nni_experiment(self):
        exp = Experiment('local')
        # ------------------------------- Do Not Change ----------------------------------
        # Set id of the experiment, to prevent the duplicate experiment in nni framework.
        exp.id = ''.join([str(ord(c)) for c in [k for k in str(datetime.datetime.now()) if k in '0123456789']])
        exp.config.experiment_name = "{},{}-way,{}-shot,{}-to-{}".format(self.opt.metric, 
                self.opt.way_num, self.opt.shot_num, 
                self.opt.source_domain, self.opt.target_domain)
        exp.config.search_space = self.paras_space
        exp.config.trial_command = 'python executor.py --mode=tune --argspath={}'.format(str(self.now_args_json))
        exp.config.trial_code_directory = '.'
        # ------------------------------- Do not Change ----------------------------------
        # ------------------------------- Can Be Modified to Change nni Configuration ----------------------------------
        # Set how many gpu to be used
        exp.config.trial_gpu_number = 1
        # Set how many experiment concurrently to be run
        exp.config.trial_concurrency = 1
        # Set early stop method to save time, see: https://nni.readthedocs.io/zh/stable/hpo/assessors.html
        exp.config.assessor.name = 'Medianstop'
        # Overall experiments to be run
        exp.config.max_trial_number = 50
        exp.config.max_experiment_duration = '24h'
        exp.config.tuner.name = 'Anneal'
        exp.config.tuner.class_args = {
            'optimize_mode': 'maximize'
        }
        # You can set training service as you wish(local or remote), see: https://nni.readthedocs.io/en/stable/experiment/training_service/overview.html
        exp.config.training_service.platform = 'local'
        exp.config.training_service.use_active_gpu = False
        # Decide how many experiment run on the one GPU
        exp.config.training_service.max_trial_number_per_gpu = 2
        # Set network port to show visualization results, see: https://nni.readthedocs.io/zh/stable/experiment/web_portal/web_portal.html
        return exp.run(8082)
        # ------------------------------- Can Be Modified to Change nni Configuration ----------------------------------
```
## TSECS (refer to'./model/TSF.py')
- TSE
```shell
python run_script_experimetal.py --gpu=GIDS --metric=TSFL
```
- TSECS with dynamic number of clusters
```shell
python run_script_experimetal.py --gpu= --metric=TSFL_BroadcastV6
```
```
- TSECS with cross-attention 
```shell
python run_script_experimetal.py --gpu=GIDS --metric=TSFL_BroadcastV10
```
