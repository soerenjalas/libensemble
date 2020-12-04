import numpy as np


def model_builder(theta, x, fevals, failures=None):
    assert theta.ndim == 2
    assert x.ndim == 2

    if failures is None:
        failures = np.isnan(fevals)

    p = x.shape[0]
    n = theta.shape[0]

    thetas_stacked = np.repeat(theta, repeats=p, axis=0)
    xs_stacked = np.tile(x.astype(float), (n, 1))

    y = fevals.flatten()
    yfail = failures.flatten()

    X = np.hstack((np.atleast_2d(np.ones(n * p)).T, thetas_stacked, xs_stacked))

    y = y[~yfail]
    X = X[~yfail]

    nug = 10 ** (-8) * np.diag(np.ones(X.shape[1]))
    nug[0, 0] = 0.

    beta = np.linalg.solve(X.T @ X + nug, X.T @ y)

    err = X @ beta - y
    var = err.T @ err / ((n * p) - beta.shape[0] - 1)

    model = {
        'beta': beta,
        'xinput': x,
        'X': X,
        'var': var}

    return model


def model_predict(model, theta):
    assert theta.ndim == 2

    beta = model['beta']
    X = model['X']
    xs = model['xinput']
    var = model['var']

    p = xs.shape[0]
    n = theta.shape[0]

    thetas_stacked = np.repeat(theta, repeats=p, axis=0)
    xs_stacked = np.tile(xs.astype(float), (n, 1))

    nug = 10 ** (-8) * np.diag(np.ones(X.shape[1]))
    nug[0, 0] = 0.

    Xnew = np.hstack((np.atleast_2d(np.ones(n * p)).T, thetas_stacked, xs_stacked))

    ynew = (Xnew @ beta).reshape((n, p))
    varnew = np.diag(var * (1 + Xnew @ np.linalg.solve(X.T @ X + nug, Xnew.T))).reshape((n, p))

    return ynew, varnew


def select_next_theta(model, theta, n_explore_theta, step_add_theta):
    xs = model['xinput']
    var = model['var']

    p = xs.shape[0]
    n = theta.shape[0]

    thetas_stacked = np.repeat(theta, repeats=p, axis=0)
    xs_stacked = np.tile(xs.astype(float), (n, 1))

    X = np.hstack((np.atleast_2d(np.ones(n * p)).T, thetas_stacked, xs_stacked))

    nug = 10 ** (-8) * np.diag(np.ones(X.shape[1]))
    nug[0, 0] = 0.

    predvar = np.diag(var * (1 + X @ np.linalg.solve(X.T @ X + nug, X.T)))

    stop_flag = False
    if np.max(predvar) < var * (1 + 10 ** (-2)):
        stop_flag = True
        theta_choose = None
    else:
        maxvarind = np.argmax(predvar)
        maxvartheta = thetas_stacked[maxvarind]

        thetanew = maxvartheta + np.sqrt(predvar[maxvarind]) *\
            np.random.normal(size=(n_explore_theta, theta.shape[1]))
        thetasnew_stacked = np.repeat(thetanew, repeats=p, axis=0)
        xsnew_stacked = np.tile(xs.astype(float), (n_explore_theta, 1))

        Xnew = np.hstack((np.atleast_2d(np.ones(n_explore_theta * p)).T, thetasnew_stacked, xsnew_stacked))
        predvarnew = np.diag(var * (1 + Xnew @ np.linalg.solve(X.T @ X + nug, Xnew.T))).reshape(n_explore_theta, p)

        choose_ind = np.argsort(predvarnew.sum(axis=1))[-step_add_theta:]
        theta_choose = thetanew[choose_ind]

    return theta_choose, stop_flag


def obviate_pend_thetas(model, theta, data_status, critthres=0.01, n_keep=1):
    mofnew = 0 + (data_status > -0.5)
    wheretheta = np.where(np.sum(mofnew*(data_status < 0.5), 1) > 0.5)[0]

    var = model['var']

    _, predvar = model_predict(model, theta)

    critvals = predvar.mean(axis=1)

    critflag = critvals < var * (1 + critthres)

    if critflag.all():
        critinds = np.argsort(critvals)
        critflag[critinds[:(-n_keep)]] = False

    critflag[~wheretheta] = False
    r_obviate = [np.array(wheretheta[np.where(critflag)])]
    return r_obviate
