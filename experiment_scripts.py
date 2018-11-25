import numpy as np
import random

manual_seed = 1338


def experiment(learner, criterion, model, model_kwargs, train_set,
               initially_labeled,
               test_loader,
               val_loader=None,
               rounds=5,
               iters=100,
               inital_training_epochs=100,
               re_training_epochs=100,
               hard=False,
               temp=0,
               print_freq=10,
               plot=True,
               plot_densfct=None):
    """
    rounds: take mean and std over rounds
    iters: total queries made -> each round will query iters-initial_labeled new instances
    inital_training_epochs: how may iterations of finetuning on initial labeled pool
    re_training_epochs: 5 # how may iterations of finetuning after each query
    hard = False: reset model weights afer each query
    temp: temperature for sampling (0 = max greedy)
    """

    random.seed(manual_seed)

    if type(initially_labeled) == int:
        n = initially_labeled
        initially_labeled = []
        for r in range(rounds):
            i = random.sample(range(len(train_set)), n)
            initially_labeled.append(i)

    else:
        assert(len(initially_labeled) == rounds)

    accs = np.zeros((rounds, iters))
    losses = np.zeros((rounds, iters))
    learners = []

    for r in range(rounds):
        kwargs = {
            'criterion': criterion,
            'model': model(**model_kwargs),
            'dataset': train_set,
            'val_loader': val_loader,
            'L_indices': initially_labeled[r],
            'verbose': False,
            'device': 'cpu'
        }

        l = learner(**kwargs)
        l.model.reset_parameters()

        if len(l.L) == 0:  # select the first instance
            inst = l.select(temp=temp)
            l.label(inst)

        init_labeled = len(l.L)

        # intitial training for warm start
        if hard==False:
            # does not make sense when model is resetted after each query
            l.re_train(epochs=inital_training_epochs, hard=False)

        print('-' * 5, 'Starting round {}'.format(r), '-' * 5)
        for i in range(init_labeled, iters + 1):
            if hard:
                l.model = model(**model_kwargs)
            l.re_train(epochs=re_training_epochs, hard=False)

            acc, loss = l.model.test(test_loader, verbose=False)
            if i % print_freq == 0 or i == (iters) or i == init_labeled:
                print('Iteration: {} | Total datapoints: {} | Val Acc: {:.4f} | Val loss {:.4f} '.format(
                    i, len(l.L), acc, loss))
                if plot:
                    l.plot(plot_densfct)

            inst = l.select(temp=temp)
            l.label(inst)
            accs[r, i - 1] = acc
            losses[r, i - 1] = loss

        learners.append(l)

    # trim leading 0's from accs and losses
    accs = accs[:, init_labeled - 1:]
    losses = losses[:, init_labeled - 1:]

    mean, std = accs.mean(axis=0), accs.std(axis=0)
    return learners, accs, losses, mean, std
