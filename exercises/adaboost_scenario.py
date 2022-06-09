import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
# from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    meta_learner = AdaBoost(wl=DecisionStump, iterations=n_learners)
    meta_learner.fit(train_X, train_y)

    fig = go.Figure(layout=go.Layout(
        title = 'Train and Test Errors as a Function of Number of Learners'),
    )
    x_values = list(range(1, n_learners))
    y_values_train = [meta_learner.partial_loss(train_X, train_y, t) for t in range(1, n_learners)]
    y_values_test = [meta_learner.partial_loss(test_X, test_y, t) for t in range(1, n_learners)]
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values_train, marker=dict(color='blue'), name='Train Set'),
    ) 
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values_test, marker=dict(color='green'), name='Test Set'),
    )
    fig.update_xaxes(title='Number of Learners').update_yaxes(title='Errors')
    fig.show()
    
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in T],
                        horizontal_spacing = 0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([
            decision_surface(lambda x: meta_learner.partial_predict(x, t), lims[0], lims[1], showscale=False),
            go.Scatter(
                x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                marker=dict(color=test_y, colorscale=[custom[0], custom[-1]], 
                line=dict(color="black", width=1))
            )
        ], 
        rows=(i//2) + 1,
        cols=(i%2)+1
        )
    fig.update_layout(title=rf"$\textbf{{Decision Boundaries by Number of Learners}}$", margin=dict(t=100))\
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_t = None
    best_value = None
    for t in range(1, n_learners):
        loss = meta_learner.partial_loss(test_X, test_y, t) 
        if not best_value or loss < best_value:
            best_t, best_value = t, loss
    acc = accuracy(test_y, meta_learner.partial_predict(test_X, best_t))
    fig = go.Figure(layout=go.Layout(
        title = f'Decision Boundry for Ensemble of size {best_t} with Accuracy {acc}',
    ))
    fig.add_traces([
        decision_surface(lambda x: meta_learner.partial_predict(x, best_t), lims[0], lims[1], showscale=False),
        go.Scatter(
            x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
            marker=dict(color=test_y, colorscale=[custom[0], custom[-1]], 
            line=dict(color="black", width=1))
        )
    ])
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure(layout=go.Layout(title='Decision Surface with Weighted Dots of Training Set'))
    fig.add_traces([
        decision_surface(meta_learner.predict, lims[0], lims[1], showscale=False),
        go.Scatter(
            x=train_X[:,0], y=train_X[:,1], mode="markers", showlegend=False,
            marker=dict(
                color=train_y, colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1),
                size=meta_learner.D_ / np.max(meta_learner.D_) * 5
            )
        ) 
    ])
    fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    for noise in (0, 0.4):
        fit_and_evaluate_adaboost(noise=noise)