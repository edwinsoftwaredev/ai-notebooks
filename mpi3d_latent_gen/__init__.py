from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch
import wandb

from mpi3d_latent_gen import demo
from mpi3d_latent_gen.functions import train_model

latent_dim = 7

config = {
    'model_config': {
        'latent_dim': latent_dim,

        'encoder_output_shape': (512, 4, 4),
        
        'kl_cost_annealing_time': 1,

        # DCGAN architecture
        'encoder': [
            {'conv2d': {'in': 3, 'out': 64, 'kernel': tune.choice([4]), 'padding': 1, 'stride': 2, 'bias': True}}, # <-- ((I + 2P - K) / S) + 1 = 32
            {'leakyRelu': {'nslope': 0.2, 'inplace': True}},
            
            {'conv2d': {'in': 64, 'out': 128, 'kernel': 4, 'padding': 1, 'stride': 2, 'bias': True}}, # 16
            # {'layerNorm': { 'shape': [128, 16, 16], 'bias': True }},
            {'leakyRelu': {'nslope': 0.2, 'inplace': True}},

            {'conv2d': {'in': 128, 'out': 256, 'kernel': 4, 'padding': 1, 'stride': 2, 'bias': True}}, # 8
            # {'layerNorm': { 'shape': [256, 8, 8], 'bias': True }},
            {'leakyRelu': {'nslope': 0.2, 'inplace': True}},

            {'conv2d': {'in': 256, 'out': 512, 'kernel': 4, 'padding': 1, 'stride': 2, 'bias': True}}, # 4
            # {'layerNorm': { 'shape': [512, 4, 4], 'bias': True }},
            {'leakyRelu': {'nslope': 0.2, 'inplace': True}},

            # {'conv2d': {'in': 512, 'out': 1, 'kernel': 4, 'padding': 0, 'stride': 1, 'bias': False}}, # 1
            # {'sigmoid': True}
            # {'adaptativeAvgPool2d': {'output_size': 1}} # 1
        ],

        'decoder': [
            {'convT2d': {'in': latent_dim, 'out': 512, 'kernel': tune.choice([4]), 'stride': 1, 'padding': 0, 'bias': True}}, # (I − 1)*S − 2*P + (K − 1) + output_padding + 1 = 4
            # {'batchNorm2d': {'in': 512, 'bias': True }},
            {'relu': {'inplace': True}},

            {'convT2d': {'in': 512, 'out': 256, 'kernel': 4, 'stride': 2, 'padding': 1, 'bias': True}}, # 8
            # {'batchNorm2d': {'in': 256, 'bias': True }},
            {'relu': {'inplace': True}},

            {'convT2d': {'in': 256, 'out': 128, 'kernel': 4, 'stride': 2, 'padding': 1, 'bias': True}}, # 16
            # {'batchNorm2d': {'in': 128, 'bias': True }},
            {'relu': {'inplace': True}},

            {'convT2d': {'in': 128, 'out': 64, 'kernel': 4, 'stride': 2, 'padding': 1, 'bias': True}}, # 32
            # {'batchNorm2d': {'in': 64, 'bias': True }},
            {'relu': {'inplace': True}},

            {'convT2d': {'in': 64, 'out': 3, 'kernel': 4, 'stride': 2, 'padding': 1, 'bias': True}}, # 64
            # {'tanh': True},
        ]
    },

    'lr': tune.choice([1e-4]),
    'batch_size': tune.choice([64]),
    'schdlr_patience': 3,
    'schdlr_factor': 0.5,
    'epochs': 200,
    'num_trials': 1,
}

scheduler = ASHAScheduler(
    time_attr='training_iteration',
    max_t=config['epochs'],
    grace_period=1,
    reduction_factor=2
)

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(train_model),
        resources={ 'cpu': 4, 'gpu': 1, 'accelerator_type:P100': 1 }
    ),

    tune_config=tune.TuneConfig(
        metric='test_loss',
        mode='min',
        scheduler=scheduler,
        num_samples=config['num_trials']
    ),

    param_space=config,

    run_config=tune.RunConfig(
        storage_path="/kaggle/working",
        name="mpi3d"
    )
)

torch.backends.cudnn.benchmark = True

results = tuner.fit()

best_result = results.get_best_result('test_loss', 'min')

print(f'Best trial config: { best_result.config }')
print(f'Best trial final validation loss: { best_result.metrics["test_loss"] }')

image = demo(best_result, latent_dim)
wandb.init(project='mpi3d_latent_gen', group='experiment_1')
wandb.log({"examples": [wandb.Image(image)]})
wandb.finish()