from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    origin_mu = 10
    X = np.random.normal(origin_mu, 1, size=1000)
    uni = UnivariateGaussian().fit(X)
    print((uni.mu, uni.var))

    # Question 2 - Empirically showing sample mean is consistent
    sizes = np.arange(10, 1010, step=10)
    fixed_by_size = np.vectorize(lambda size: np.abs(uni.fit(X[:size]).mu-origin_mu))(sizes)
    # print(fixed_by_size)
    fig = go.Figure(layout_title_text='distance from mean by sample size',
                    data=[go.Bar(x=sizes, y=fixed_by_size)])
    fig.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    fig = go.Figure(data=go.Scatter(x=X, y=uni.pdf(X), mode='markers', marker=dict(color="black"), showlegend=False)) 
    #.update_layout(title_text=r"$\text{(1) Generating Data From Model}$", height=300)
    fig.show()
    
    #fig = go.Figure(data=[go.Scatter(x=X, y=uni.pdf(X))])
    #fig.show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
