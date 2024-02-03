

problem_configuration = {
    'NAS101': {
        'maxEvals': 3000,
        'dataset': 'CIFAR-10',
        'epoch': '36'
    },
    'NAS201-C10': {
        'maxEvals': 3000,
        'dataset': 'CIFAR-10',
        'epoch': '12'
    },
    'NAS201-C100': {
        'maxEvals': 3000,
        'dataset': 'CIFAR-100',
        'epoch': '12'
    },
    'NAS201-IN16': {
        'maxEvals': 3000,
        'dataset': 'ImageNet16-120',
        'epoch': '12'
    }
}


def get_algorithm(algorithm_name, **kwargs):
    if algorithm_name == 'NSGA-II':
        from algorithms import NSGAII
        return NSGAII()
    else:
        raise ValueError(f'Not supporting this algorithm - {algorithm_name}.')


if __name__ == '__main__':
    pass