smplifyx_stages = {
    'Stage 1': {
        'num_iter': 10,
    },
    'Stage 2': {
        'num_iter': 1,
    }
}

smplifyx_opt_config = {
    'type': 'LBFGS',
    'max_iter': 20,
    'lr': 1e-2,
    'line_search_fn': 'strong_wolfe'
}
