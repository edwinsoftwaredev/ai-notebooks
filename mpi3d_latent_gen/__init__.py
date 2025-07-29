from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch
import wandb

from mpi3d_latent_gen import demo
from mpi3d_latent_gen.functions import train_model

latent_dim = 10

config = {
    'model_config': {
        'latent_dim': latent_dim,
        
        'encoder_output_shape': (256, 5, 5),
        
        'kl_cost_annealing_time': 75,

        'encoder': {
            'l01': {
                'conv2d': {
                    'in': 3,
                    'out': 32,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([1]),
                    'stride': tune.choice([1])   # <-- ((I + 2P - K) / S) + 1 = 64
                }
            },
            'l02': {
                'conv2d': {
                    'in': 32,
                    'out': 64,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([1])  # 62
                }
            },
            'l03': {
                'conv2d': {
                    'in': 64,
                    'out': 64,
                    'kernel': tune.choice([4]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([2])  # 30
                }
            },
            'l04': {
                'conv2d': {
                    'in': 64,
                    'out': 128,
                    'kernel': tune.choice([4]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([2])  # 14
                }
            },
            'l05': {
                'conv2d': {
                    'in': 128,
                    'out': 256,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([1])  # 12
                }
            },
            'l06': {
                'conv2d': {
                    'in': 256,
                    'out': 256,
                    'kernel': tune.choice([4]),
                    'padding': tune.choice([0]),
                    'stride': tune.choice([2])  # 5
                }
            },
        },
        
        'decoder': {
            'l01': {
                'convT2d': {
                    'in': 256,
                    'out': 256,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([1]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # (I − 1) * S − 2*P + (K − 1) + output_padding + 1 = 5
                }
            },
            'l02': {
                'convT2d': {
                    'in': 256,
                    'out': 256,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([1]),
                    'stride': tune.choice([2]) # 12
                }
            },
            'l03': {
                'convT2d': {
                    'in': 256,
                    'out': 256,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([1]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # 12
                }
            },
            'l04': {
                'convT2d': {
                    'in': 256,
                    'out': 128,
                    'kernel': tune.choice([4]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([1]),
                    'stride': tune.choice([2]) # 27
                }
            },
            'l05': {
                'convT2d': {
                    'in': 128,
                    'out': 128,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([1]),
                    'stride': tune.choice([2]) # 56
                }
            },
            'l06': {
                'convT2d': {
                    'in': 128,
                    'out': 64,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([1]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # 56
                }
            },
            'l07': {
                'convT2d': {
                    'in': 64,
                    'out': 32,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([1]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # 56
                }
            },
            'l08': {
                'convT2d': {
                    'in': 32,
                    'out': 32,
                    'kernel': tune.choice([4]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # 59
                }
            },
            'l09': {
                'convT2d': {
                    'in': 32,
                    'out': 16,
                    'kernel': tune.choice([4]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # 62
                }
            },
            'l10': {
                'convT2d': {
                    'in': 16,
                    'out': 3,
                    'kernel': tune.choice([3]),
                    'padding': tune.choice([0]),
                    'out_padding': tune.choice([0]),
                    'stride': tune.choice([1]) # 64
                },
                'output': True
            }
        }
    },

    'lr': tune.choice([1e-3]),
    'batch_size': tune.choice([128]),
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