import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from functions import train_model

config = {
    'model_config': {
        'l1_c': tune.choice([32]),
        'l1_cnk': tune.choice([3]),
        'l1_pk': tune.choice([3]),
        'l1_ps': tune.choice([1]),
    
        'l2_c': tune.choice([64]),
        'l2_cnk': tune.choice([3]),
        'l2_pk': tune.choice([3]),
        'l2_ps': tune.choice([1]),
    
        'l3_c': tune.choice([128]),
        'l3_cnk': tune.choice([3]),
        'l3_pk': tune.choice([3]),
        'l3_ps': tune.choice([1]),
    
        'l4_c': tune.choice([256]),
        'l4_cnk': tune.choice([3]),
        'l4_pk': tune.choice([3]),
        'l4_ps': tune.choice([1]),
    
        'l5_c': tune.choice([512]),
        'l5_cnk': tune.choice([3]),
        'l5_pk': tune.choice([3]),
        'l5_ps': tune.choice([1]),

        # 'l6_c': tune.choice([256]),
        # 'l6_cnk': tune.choice([3]),
        # 'l6_pk': tune.choice([3]),
        # 'l6_ps': tune.choice([1]),

        # 'l7_c': tune.choice([256]),
        # 'l7_cnk': tune.choice([3]),
        # 'l7_pk': tune.choice([3]),
        # 'l7_ps': tune.choice([1]),

        # 'l8_c': tune.choice([512]),
        # 'l8_cnk': tune.choice([3]),
        # 'l8_pk': tune.choice([3]),
        # 'l8_ps': tune.choice([1]),

        # 'l9_c': tune.choice([512]),
        # 'l9_cnk': tune.choice([3]),
        # 'l9_pk': tune.choice([3]),
        # 'l9_ps': tune.choice([1]),

        # 'l10_c': tune.choice([512]),
        # 'l10_cnk': tune.choice([3]),
        # 'l10_pk': tune.choice([3]),
        # 'l10_ps': tune.choice([1]),
    },

    'lr': tune.choice([1e-2]),
    'weight_decay': tune.choice([5e-4]),
    'schdler_patience': tune.choice([3]),
    'schdler_factor': tune.choice([0.5]),
    'batch_size': tune.choice([64]),
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