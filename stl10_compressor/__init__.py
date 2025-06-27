from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch

from stl10_compressor.functions import train_model

config = {
    'model_config': {
        'encoder': {
            'l01': {
                'conv2d': {
                    'in': 3,
                    'out': 32,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([1]),
                    'stride': tune.choice([1])   # <-- ((I + 2P - K) / S) + 1 = 96
                }
            },
            'l02': {
                'conv2d': {
                    'in': 32,
                    'out': 64,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([1])  # 94
                }
            },
            'l03': {
                'conv2d': {
                    'in': 64,
                    'out': 128,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([1])  # 92
                }
            },
            'l04': {
                'conv2d': {
                    'in': 128,
                    'out': 256,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([2])  # 45
                }
            }
        },

        'code': {
            'l01': {
                'conv2d': {
                    'in': 256,
                    'out': 512,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([2])  # 22
                }
            }
        },
        
        'decoder': {
            'l01': {
                'convT2d': {
                    'in': 512,
                    'out': 128,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([2]) # (I − 1) * S − 2*P + (K − 1) + output_padding + 1 = 45
                }
            },
            'l02': {
                'convT2d': {
                    'in': 128,
                    'out': 64,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([1]),
                    'stride': tune.choice([2]) # 92
                }
            },
            'l03': {
                'convT2d': {
                    'in': 64,
                    'out': 32,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # 94
                }
            },
            'l04': {
                'convT2d': {
                    'in': 32,
                    'out': 3,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # 96
                },
                'output': True
            }
        }
    },

    'lr': tune.choice([1e-3]),
    'batch_size': tune.choice([512]),
    'weight_decay': 5e-4,
    'schdlr_patience': 3,
    'schdlr_factor': 0.5,
    'epochs': 100,
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
        name="stl10"
    )
)

torch.backends.cudnn.benchmark = True

results = tuner.fit()

best_result = results.get_best_result('test_loss', 'min')

print(f'Best trial config: { best_result.config }')
print(f'Best trial final validation loss: { best_result.metrics["test_loss"] }')
print(f'Best trial final validation accuracy: { best_result.metrics["test_accuracy"] }')