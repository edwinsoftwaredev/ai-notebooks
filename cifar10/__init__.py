import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from functions import train_model

config = {
    'model_config': {
        'l01': {
            'conv2d': {
                'channels': tune.choice([32]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            },
            'dropout': {
                'prob': 0.3
            },
            # 'maxpool': {
            #     'kernel': tune.choice([3]),
            #     'padding': tune.choice([1]),
            #     'stride': tune.choice([1]),
            # }
        },
        'l02': {
            'conv2d': {
                'channels': tune.choice([32]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            },
        },
        'l03': {
            'conv2d': {
                'channels': tune.choice([32]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([0]),
                'stride': tune.choice([1])
            },
            'dropout': {
                'prob': 0.3
            },
        },
        'l04': {
            'conv2d': {
                'channels': tune.choice([64]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([0]),
                'stride': tune.choice([1])
            },
        },
        'l05': {
            'conv2d': {
                'channels': tune.choice([64]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            },
        },
        'l06': {
            'conv2d': {
                'channels': tune.choice([64]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([2])
            },
        },
        'l07': {
            'conv2d': {
                'channels': tune.choice([128]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            },
            'dropout': {
                'prob': 0.2
            },
        },
        'l08': {
            'conv2d': {
                'channels': tune.choice([128]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([0]),
                'stride': tune.choice([1])
            },
        },
        'l09': {
            'conv2d': {
                'channels': tune.choice([128]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            },
        },
        'l10': {
            'conv2d': {
                'channels': tune.choice([256]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            },
           'dropout': {
                'prob': 0.2
            }
        },
        'l11': {
            'conv2d': {
                'channels': tune.choice([256]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([0]),
                'stride': tune.choice([1])
            },
        },
        'l12': {
            'conv2d': {
                'channels': tune.choice([256]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            },
        },
        'l13': {
            'conv2d': {
                'channels': tune.choice([512]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            },
        },
        'l14': {
            'conv2d': {
                'channels': tune.choice([512]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([0]),
                'stride': tune.choice([1])
            }
        },
        'l15': {
            'conv2d': {
                'channels': tune.choice([512]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            }
        },
        'l16': {
            'conv2d': {
                'channels': tune.choice([512]),
                'kernel': tune.choice([3]),
                'padding': tune.choice([1]),
                'stride': tune.choice([1])
            }
        }
    },

    'lr': tune.choice([1e-2]),
    'batch_size': tune.choice([64]),
    'weight_decay': 1e-5,
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
        resources={ 'cpu': 2, 'gpu': 1, 'accelerator_type:T4': 1 }
    ),

    tune_config=tune.TuneConfig(
        metric='test_loss',
        mode='min',
        scheduler=scheduler,
        num_samples=config['num_trials']
    ),

    param_space=config
)

torch.backends.cudnn.benchmark = True

results = tuner.fit()

best_result = results.get_best_result('test_loss', 'min')

print(f'Best trial config: { best_result.config }')
print(f'Best trial final validation loss: { best_result.metrics["test_loss"] }')
print(f'Best trial final validation accuracy: { best_result.metrics["test_accuracy"] }')