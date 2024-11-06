import torch


class RBM:
    """Restricted Boltzmann Machine (RBM) 

    A Restricted Boltzmann Machine with binary visible units and
    binary hidden units. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Read more in the :ref:`User Guide <rbm>`.

    Parameters
    ----------
    n_components : int, default=256
        Number of binary hidden units.

    learning_rate : float, default=0.1
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, default=10
        Number of examples per minibatch.

    n_iter : int, default=10
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, default=0
        The verbosity level. The default, zero, means silent mode. Range
        of values is [0, inf].

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for:

        - Gibbs sampling from visible and hidden layers.

        - Initializing components, sampling from layers during fit.

        - Corrupting the data when scoring samples.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    intercept_hidden_ : array-like of shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like of shape (n_features,)
        Biases of the visible units.

    components_ : array-like of shape (n_components, n_features)
        Weight matrix, where `n_features` is the number of
        visible units and `n_components` is the number of hidden units.

    h_samples_ : array-like of shape (batch_size, n_components)
        Hidden Activation sampled from the model distribution,
        where `batch_size` is the number of examples per minibatch and
        `n_components` is the number of hidden units.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008

    Examples
    --------

    >>> import torch
    >>> from rbm import RBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = RBM(n_components=2)
    >>> model.fit(X)
    RBM(n_components=2)

    For a more detailed example usage, see
    :ref:`sphx_glr_auto_examples_neural_networks_plot_rbm_logistic_classification.py`.
    """

    def __init__(self, n_components: int=256, learning_rate: float=0.1, batch_size: int=10,
                 n_iter: int=10, verbose: int=0, random_state: int=None):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state