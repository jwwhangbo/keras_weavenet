from typing import Callable, List, Union, Sequence, Optional, Dict
from tensorflow import keras
from collections.abc import Sequence
from keras.layers import Input, Dense, BatchNormalization, Dropout, Reshape, Softmax, Conv1D, Flatten # type: ignore
import tensorflow as tf

class WeaveLayer(keras.layers.Layer):
    """This class implements the core Weave convolution from the
    Google graph convolution paper [1]_

    This model contains atom features and bond features
    separately.Here, bond features are also called pair features.
    There are 2 types of transformation, atom->atom, atom->pair,
    pair->atom, pair->pair that this model implements.

    Examples
    --------
    This layer expects 4 inputs in a list of the form `[atom_features,
    pair_features, pair_split, atom_to_pair]`. We'll walk through the structure
    of these inputs. Let's start with some basic definitions.

    >>> import deepchem as dc
    >>> import numpy as np

    Suppose you have a batch of molecules

    >>> smiles = ["CCC", "C"]

    Note that there are 4 atoms in total in this system. This layer expects its
    input molecules to be batched together.

    >>> total_n_atoms = 4

    Let's suppose that we have a featurizer that computes `n_atom_feat` features
    per atom.

    >>> n_atom_feat = 75

    Then conceptually, `atom_feat` is the array of shape `(total_n_atoms,
    n_atom_feat)` of atomic features. For simplicity, let's just go with a
    random such matrix.

    >>> atom_feat = np.random.rand(total_n_atoms, n_atom_feat)

    Let's suppose we have `n_pair_feat` pairwise features

    >>> n_pair_feat = 14

    For each molecule, we compute a matrix of shape `(n_atoms*n_atoms,
    n_pair_feat)` of pairwise features for each pair of atoms in the molecule.
    Let's construct this conceptually for our example.

    >>> pair_feat = [np.random.rand(3*3, n_pair_feat), np.random.rand(1*1, n_pair_feat)]
    >>> pair_feat = np.concatenate(pair_feat, axis=0)
    >>> pair_feat.shape
    (10, 14)

    `pair_split` is an index into `pair_feat` which tells us which atom each row belongs to. In our case, we hve

    >>> pair_split = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])

    That is, the first 9 entries belong to "CCC" and the last entry to "C". The
    final entry `atom_to_pair` goes in a little more in-depth than `pair_split`
    and tells us the precise pair each pair feature belongs to. In our case

    >>> atom_to_pair = np.array([[0, 0],
    ...                          [0, 1],
    ...                          [0, 2],
    ...                          [1, 0],
    ...                          [1, 1],
    ...                          [1, 2],
    ...                          [2, 0],
    ...                          [2, 1],
    ...                          [2, 2],
    ...                          [3, 3]])

    Let's now define the actual layer

    >>> layer = WeaveLayer()

    And invoke it

    >>> [A, P] = layer([atom_feat, pair_feat, pair_split, atom_to_pair])

    The weave layer produces new atom/pair features. Let's check their shapes

    >>> A = np.array(A)
    >>> A.shape
    (4, 50)
    >>> P = np.array(P)
    >>> P.shape
    (10, 50)

    The 4 is `total_num_atoms` and the 10 is the total number of pairs. Where
    does `50` come from? It's from the default arguments `n_atom_input_feat` and
    `n_pair_input_feat`.

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
        fingerprints." Journal of computer-aided molecular design 30.8 (2016):
        595-608.

    """
    def __init__(self,
                 n_atom_input_feat: int = 9,
                 n_pair_input_feat: int = 2,
                 n_atom_output_feat: int = 50,
                 n_pair_output_feat: int = 50,
                 max_n_atoms: int = 50,
                 max_n_pairs: int = 50,
                 n_hidden_AA: int = 50,
                 n_hidden_PA: int = 50,
                 n_hidden_AP: int = 50,
                 n_hidden_PP: int = 50,
                 update_pair: bool = True,
                 init: Union[str, Callable] = 'glorot_uniform',
                 activation: Callable = keras.activations.relu,
                 batch_normalize: bool = True,
                 batch_normalize_kwargs: Dict = {"renorm": True},
                 **kwargs):
        """
        Parameters
        ----------
        n_atom_input_feat: int, optional (default 75)
            Number of features for each atom in input.
        n_pair_input_feat: int, optional (default 14)
            Number of features for each pair of atoms in input.
        n_atom_output_feat: int, optional (default 50)
            Number of features for each atom in output.
        n_pair_output_feat: int, optional (default 50)
            Number of features for each pair of atoms in output.
        n_hidden_AA: int, optional (default 50)
            Number of units(convolution depths) in corresponding hidden layer
        n_hidden_PA: int, optional (default 50)
            Number of units(convolution depths) in corresponding hidden layer
        n_hidden_AP: int, optional (default 50)
            Number of units(convolution depths) in corresponding hidden layer
        n_hidden_PP: int, optional (default 50)
            Number of units(convolution depths) in corresponding hidden layer
        update_pair: bool, optional (default True)
            Whether to calculate for pair features,
            could be turned off for last layer
        init: str, optional (default 'glorot_uniform')
            Weight initialization for filters.
        activation: str, optional (default 'relu')
            Activation function applied
        batch_normalize: bool, optional (default True)
            If this is turned on, apply batch normalization before applying
            activation functions on convolutional layers.
        batch_normalize_kwargs: Dict, optional (default `{renorm=True}`)
            Batch normalization is a complex layer which has many potential
            argumentswhich change behavior. This layer accepts user-defined
            parameters which are passed to all `BatchNormalization` layers in
            `WeaveModel`, `WeaveLayer`, and `WeaveGather`.
        """
        super(WeaveLayer, self).__init__(**kwargs)
        self.init = init  # Set weight initialization
        self.activation = activation  # Get activations
        self.update_pair = update_pair  # last weave layer does not need to update
        self.n_hidden_AA = n_hidden_AA
        self.n_hidden_PA = n_hidden_PA
        self.n_hidden_AP = n_hidden_AP
        self.n_hidden_PP = n_hidden_PP
        self.n_hidden_A = n_hidden_AA
        self.n_hidden_P = n_hidden_AP
        self.batch_normalize = batch_normalize
        self.batch_normalize_kwargs = batch_normalize_kwargs

        self.n_atom_input_feat = n_atom_input_feat
        self.n_pair_input_feat = n_pair_input_feat
        self.n_atom_output_feat = n_atom_output_feat
        self.n_pair_output_feat = n_pair_output_feat
        # self.output_feat = n_atom_output_feat + n_pair_output_feat
        self.max_n_atoms = max_n_atoms
        self.max_n_pairs = max_n_pairs
        self.W_AP, self.b_AP, self.W_PP, self.b_PP, self.W_P, self.b_P = None, None, None, None, None, None

    def get_config(self) -> Dict:
        """Returns config dictionary for this layer."""
        config = super(WeaveLayer, self).get_config()
        config['n_atom_input_feat'] = self.n_atom_input_feat
        config['n_pair_input_feat'] = self.n_pair_input_feat
        config['n_atom_output_feat'] = self.n_atom_output_feat
        config['n_pair_output_feat'] = self.n_pair_output_feat
        config['n_hidden_AA'] = self.n_hidden_AA
        config['n_hidden_PA'] = self.n_hidden_PA
        config['n_hidden_AP'] = self.n_hidden_AP
        config['n_hidden_PP'] = self.n_hidden_PP
        config['batch_normalize'] = self.batch_normalize
        config['batch_normalize_kwargs'] = self.batch_normalize_kwargs
        config['update_pair'] = self.update_pair
        config['init'] = self.init
        config['activation'] = self.activation
        return config

    def build(self, input_shape):
        """ Construct internal trainable weights.

        Parameters
        ----------
        input_shape: tuple
            Ignored since we don't need the input shape to create internal weights.
        """

        def init(input_shape):
            return self.add_weight(name='kernel',
                                   shape=(input_shape[0],input_shape[1]),
                                   initializer=self.init,
                                   trainable=True)

        self.W_AA = init([self.n_atom_input_feat, self.n_atom_input_feat])
        self.b_AA = tf.zeros(shape=[
            self.n_atom_input_feat,
        ])
        self.AA_bn = BatchNormalization(**self.batch_normalize_kwargs)

        self.W_PA = init([self.n_pair_input_feat, self.n_pair_input_feat])
        self.b_PA = tf.zeros(shape=[
            self.n_pair_input_feat,
        ])
        self.PA_bn = BatchNormalization(**self.batch_normalize_kwargs)

        self.W_A = init([self.n_atom_input_feat + self.n_pair_input_feat, self.n_atom_output_feat])
        self.b_A = tf.zeros(shape=[
            self.n_atom_output_feat,
        ])
        self.A_bn = BatchNormalization(**self.batch_normalize_kwargs)

        if self.update_pair:
            self.W_AP = init([self.n_atom_input_feat, self.n_atom_input_feat])
            self.b_AP = tf.zeros(shape=[
                self.n_atom_input_feat,
            ])
            self.AP_bn = BatchNormalization(**self.batch_normalize_kwargs)

            self.W_PP = init([self.n_pair_input_feat, self.n_pair_input_feat])
            self.b_PP = tf.zeros(shape=[
                self.n_pair_input_feat,
            ])
            self.PP_bn = BatchNormalization(**self.batch_normalize_kwargs)

            self.W_P = init([self.n_atom_input_feat + self.n_pair_input_feat, self.n_pair_output_feat])
            self.b_P = tf.zeros(shape=[
                self.n_pair_output_feat,
            ])
            self.P_bn = BatchNormalization(**self.batch_normalize_kwargs)
        self.built = True

    def call(self, inputs: List) -> List:
        """Creates weave tensors.

        Parameters
        ----------
        inputs: List
            Should contain 4 tensors [atom_features, pair_features, pair_split,
            atom_to_pair]
        """
        atom_features = inputs[0]
        pair_features = inputs[1]

        atom_to_pair = inputs[2]

        activation = self.activation

        AA = tf.matmul(atom_features, self.W_AA) + self.b_AA
        if self.batch_normalize:
            AA = self.AA_bn(AA)
        AA = activation(AA)

        # This is wrong !!!
        # PA = tf.matmul(pair_features, self.W_PA) + self.b_PA
        # if self.batch_normalize:
        #     PA = self.PA_bn(PA)
        # PA = activation(PA)

        # apply f(linear activation) to pair features
        vPA = tf.matmul(pair_features, self.W_PA) + self.b_PA
        vPA = activation(vPA)

        batch_size = tf.shape(vPA)[0]  # Assuming A has shape (N, 72, 10)

        # Step 1: Flatten the atom_to_pair matrix to create a 1D list of atom indices
        atom_indices = tf.reshape(atom_to_pair, [batch_size, -1])  # Shape: (2 * n_pairs,)

        # Step 2: Create bond indices by repeating the range of bond indices for each atom in the bond
        bond_indices = tf.repeat(tf.range(self.max_n_pairs), repeats=2)  # Shape: (2 * n_pairs,)
        bond_indices = tf.tile(tf.expand_dims(bond_indices, axis=0), [batch_size, 1])  # Shape: (N, 2 * n_pairs)


        # Step 3: Combine atom_indices and bond_indices into index pairs for scatter_nd
        indices = tf.stack([atom_indices, bond_indices], axis=2)  # Shape: (N, 2 * n_pairs, 2)

        # Step 4: Use scatter_nd to create a one-hot matrix of shape (n_atoms, n_pairs)
        one_hot_matrix = tf.scatter_nd(indices, tf.ones_like(bond_indices, dtype=tf.float32), shape=(self.max_n_atoms, self.max_n_pairs))
        # Use tf.scatter_nd to sum up bond features for each atom
        PA = tf.matmul(one_hot_matrix, vPA)  # Shape: (N, n_atoms, n_pair_input_feat)

        A = tf.matmul(tf.concat([AA, PA], axis=2), self.W_A) + self.b_A
        if self.batch_normalize:
            A = self.A_bn(A)
        A = activation(A)

        if self.update_pair:
            # Note that AP_ij and AP_ji share the same self.AP_bn batch
            # normalization
            AP = tf.matmul(
                tf.reduce_sum(tf.gather(atom_features, atom_to_pair, batch_dims=1),
                              axis=1),
                self.W_AP) + self.b_AP
            if self.batch_normalize:
                AP = self.AP_bn(AP)
            AP = activation(AP)
            # AP_ji = tf.matmul(
            #     tf.reshape(
            #         tf.gather(atom_features, tf.reverse(atom_to_pair, [1])),
            #         [-1, 2 * self.n_atom_input_feat]), self.W_AP) + self.b_AP
            # if self.batch_normalize:
            #     AP_ji = self.AP_bn(AP_ji)
            # AP_ji = activation(AP_ji)

            PP = tf.matmul(pair_features, self.W_PP) + self.b_PP
            if self.batch_normalize:
                PP = self.PP_bn(PP)
            PP = activation(PP)
            P = tf.matmul(tf.concat([AP, PP], axis=2),
                          self.W_P) + self.b_P
            if self.batch_normalize:
                P = self.P_bn(P)
            P = activation(P)
        else:
            P = pair_features

        return [A, P]
    
class WeaveGather(tf.keras.layers.Layer):
    """Implements the weave-gathering section of weave convolutions.

    Implements the gathering layer from [1]_. The weave gathering layer gathers
    per-atom features to create a molecule-level fingerprint in a weave
    convolutional network. This layer can also performs Gaussian histogram
    expansion as detailed in [1]_. Note that the gathering function here is
    simply addition as in [1]_>

    Examples
    --------
    This layer expects 2 inputs in a list of the form `[atom_features,
    pair_features]`. We'll walk through the structure
    of these inputs. Let's start with some basic definitions.

    >>> import deepchem as dc
    >>> import numpy as np

    Suppose you have a batch of molecules

    >>> smiles = ["CCC", "C"]

    Note that there are 4 atoms in total in this system. This layer expects its
    input molecules to be batched together.

    >>> total_n_atoms = 4

    Let's suppose that we have `n_atom_feat` features per atom.

    >>> n_atom_feat = 75

    Then conceptually, `atom_feat` is the array of shape `(total_n_atoms,
    n_atom_feat)` of atomic features. For simplicity, let's just go with a
    random such matrix.

    >>> atom_feat = np.random.rand(total_n_atoms, n_atom_feat)

    We then need to provide a mapping of indices to the atoms they belong to. In
    ours case this would be

    >>> atom_split = np.array([0, 0, 0, 1])

    Let's now define the actual layer

    >>> gather = WeaveGather(batch_size=2, n_input=n_atom_feat)
    >>> output_molecules = gather([atom_feat, atom_split])
    >>> len(output_molecules)
    2

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
        fingerprints." Journal of computer-aided molecular design 30.8 (2016):
        595-608.

    Note
    ----
    This class requires `tensorflow_probability` to be installed.
    """

    def __init__(self,
                 batch_size: int,
                 n_input: int = 128,
                 gaussian_expand: bool = True,
                 compress_post_gaussian_expansion: bool = False,
                 init: str = 'glorot_uniform',
                 activation: Callable = keras.activations.tanh,
                 **kwargs):
        """
        Parameters
        ----------
        batch_size: int
            number of molecules in a batch
        n_input: int, optional (default 128)
            number of features for each input molecule
        gaussian_expand: boolean, optional (default True)
            Whether to expand each dimension of atomic features by gaussian histogram
        compress_post_gaussian_expansion: bool, optional (default False)
            If True, compress the results of the Gaussian expansion back to the
            original dimensions of the input by using a linear layer with specified
            activation function. Note that this compression was not in the original
            paper, but was present in the original DeepChem implementation so is
            left present for backwards compatibility.
        init: str, optional (default 'glorot_uniform')
            Weight initialization for filters if `compress_post_gaussian_expansion`
            is True.
        activation: str, optional (default 'tanh')
            Activation function applied for filters if
            `compress_post_gaussian_expansion` is True. Should be recognizable by
            `tf.keras.activations`.
        """
        try:
            import tensorflow_probability as tfp  # noqa: F401
        except ModuleNotFoundError:
            raise ImportError(
                "This class requires tensorflow-probability to be installed.")
        super(WeaveGather, self).__init__(**kwargs)
        self.n_input = n_input
        self.batch_size = batch_size
        self.gaussian_expand = gaussian_expand
        self.compress_post_gaussian_expansion = compress_post_gaussian_expansion
        self.init = init  # Set weight initialization
        self.activation = activation  # Get activations

    def get_config(self):
        config = super(WeaveGather, self).get_config()
        config['batch_size'] = self.batch_size
        config['n_input'] = self.n_input
        config['gaussian_expand'] = self.gaussian_expand
        config['init'] = self.init
        config['activation'] = self.activation
        config[
            'compress_post_gaussian_expansion'] = self.compress_post_gaussian_expansion
        return config

    def build(self, input_shape):
        if self.compress_post_gaussian_expansion:

            def init(input_shape):
                return self.add_weight(name='kernel',
                                       shape=(input_shape[0], input_shape[1]),
                                       initializer=self.init,
                                       trainable=True)

            self.W = init([self.n_input * 11, self.n_input])
            self.b = tf.zeros(shape=[self.n_input])
        self.built = True

    def call(self, inputs: List) -> List:
        """Creates weave tensors.

        Parameters
        ----------
        inputs: List
            Should contain 2 tensors [atom_features, atom_split]

        Returns
        -------
        output_molecules: List
            Each entry in this list is of shape `(self.n_inputs,)`

        """
        outputs = inputs
        # atom_split = inputs[1]

        if self.gaussian_expand:
            outputs = self.gaussian_histogram(outputs)

        output_molecules = outputs

        if self.compress_post_gaussian_expansion:
            output_molecules = tf.matmul(outputs, self.W) + self.b
            output_molecules = self.activation(output_molecules)

        return output_molecules

    def gaussian_histogram(self, x):
        """Expands input into a set of gaussian histogram bins.

        Parameters
        ----------
        x: tf.Tensor
            Of shape `(N, n_feat)`

        Examples
        --------
        This method uses 11 bins spanning portions of a Gaussian with zero mean
        and unit standard deviation.

        >>> gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170),
        ...                         (-0.739, 0.134), (-0.468, 0.118),
        ...                         (-0.228, 0.114), (0., 0.114),
        ...                         (0.228, 0.114), (0.468, 0.118),
        ...                         (0.739, 0.134), (1.080, 0.170),
        ...                         (1.645, 0.283)]

        We construct a Gaussian at `gaussian_memberships[i][0]` with standard
        deviation `gaussian_memberships[i][1]`. Each feature in `x` is assigned
        the probability of falling in each Gaussian, and probabilities are
        normalized across the 11 different Gaussians.

        Returns
        -------
        outputs: tf.Tensor
            Of shape `(N, 11*n_feat)`
        """
        import tensorflow_probability as tfp
        batch_size = tf.shape(x)[0]
        num_atoms = tf.shape(x)[1]
        num_features = tf.shape(x)[2]
        gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170),
                                (-0.739, 0.134), (-0.468, 0.118),
                                (-0.228, 0.114), (0., 0.114), (0.228, 0.114),
                                (0.468, 0.118), (0.739, 0.134), (1.080, 0.170),
                                (1.645, 0.283)]
        dist = [
            tfp.distributions.Normal(p[0], p[1]) for p in gaussian_memberships
        ]
        dist_max = [dist[i].prob(gaussian_memberships[i][0]) for i in range(11)]
        outputs = [dist[i].prob(x) / dist_max[i] for i in range(11)]
        outputs = tf.stack(outputs, axis=-1)
        outputs = outputs / (tf.reduce_sum(outputs, axis=3, keepdims=True) + 1e-9)
        outputs = tf.reshape(outputs, [batch_size, num_atoms, num_features * 11])
        return outputs

class WeaveNet():
    def __init__(self,
                 n_tasks: int,
                 n_atom_feat: Union[int, Sequence[int]] = 10,
                 max_n_atoms: int = 72,
                 n_pair_feat: Union[int, Sequence[int]] = 2,
                 max_n_pairs: int = 78,
                 n_hidden: int = 50,
                 n_graph_feat: int = 128,
                 n_weave: int = 2,
                 fully_connected_layer_sizes: list[int] = [2000, 100],
                 conv_weight_init_stddevs: Union[float, Sequence[float]] = 0.03,
                 weight_init_stddevs: Union[float, Sequence[float]] = 0.01,
                 bias_init_consts: Union[float, Sequence[float]] = 0.0,
                 weight_decay_penalty: float = 0.0,
                 weight_decay_penalty_type: str = "l2",
                 dropouts: Union[float, Sequence[float]] = 0.25,
                 final_conv_activation_fn: Optional[callable] = keras.activations.tanh,
                 final_conv_kernel_size: int = 1,
                 activation_fns: Union[callable, Sequence[callable]] = keras.activations.relu,
                 batch_normalize: bool = True,
                 batch_normalize_kwargs: dict = {},
                 gaussian_expand: bool = True,
                 compress_post_gaussian_expansion: bool = False,
                 mode: str = "classification",
                 n_classes: int = 2,
                 batch_size: int = 100,
                 weave_kernel_initializer='glorot_uniform',
                 fully_connected_kernel_initializer='glorot_uniform',
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_atom_feat: int, optional (default 75)
            Number of features per atom. Note this is 75 by default and should be 78
            if chirality is used by `WeaveFeaturizer`.
        n_pair_feat: int, optional (default 14)
            Number of features per pair of atoms.
        n_hidden: int, optional (default 50)
            Number of units(convolution depths) in corresponding hidden layer
        n_graph_feat: int, optional (default 128)
            Number of output features for each molecule(graph)
        n_weave: int, optional (default 2)
            The number of weave layers in this model.
        fully_connected_layer_sizes: list (default `[2000, 100]`)
            The size of each dense layer in the network.  The length of
            this list determines the number of layers.
        conv_weight_init_stddevs: list or float (default 0.03)
            The standard deviation of the distribution to use for weight
            initialization of each convolutional layer. The length of this lisst
            should equal `n_weave`. Alternatively, this may be a single value instead
            of a list, in which case the same value is used for each layer.
        weight_init_stddevs: list or float (default 0.01)
            The standard deviation of the distribution to use for weight
            initialization of each fully connected layer.  The length of this list
            should equal len(layer_sizes).  Alternatively this may be a single value
            instead of a list, in which case the same value is used for every layer.
        bias_init_consts: list or float (default 0.0)
            The value to initialize the biases in each fully connected layer.  The
            length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in
            which case the same value is used for every layer.
        weight_decay_penalty: float (default 0.0)
            The magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str (default "l2")
            The type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float (default 0.25)
            The dropout probablity to use for each fully connected layer.  The length of this list
            should equal len(layer_sizes).  Alternatively this may be a single value
            instead of a list, in which case the same value is used for every layer.
        final_conv_activation_fn: Optional[ActivationFn] (default `tf.nn.tanh`)
            The Tensorflow activation funcntion to apply to the final
            convolution at the end of the weave convolutions. If `None`, then no
            activate is applied (hence linear).
        activation_fns: list or object (default `tf.nn.relu`)
            The Tensorflow activation function to apply to each fully connected layer.  The length
            of this list should equal len(layer_sizes).  Alternatively this may be a
            single value instead of a list, in which case the same value is used for
            every layer.
        batch_normalize: bool, optional (default True)
            If this is turned on, apply batch normalization before applying
            activation functions on convolutional and fully connected layers.
        batch_normalize_kwargs: Dict, optional (default `{"renorm"=True, "fused": False}`)
            Batch normalization is a complex layer which has many potential
            argumentswhich change behavior. This layer accepts user-defined
            parameters which are passed to all `BatchNormalization` layers in
            `WeaveModel`, `WeaveLayer`, and `WeaveGather`.
        gaussian_expand: boolean, optional (default True)
            Whether to expand each dimension of atomic features by gaussian
            histogram
        compress_post_gaussian_expansion: bool, optional (default False)
            If True, compress the results of the Gaussian expansion back to the
            original dimensions of the input.
        mode: str (default "classification")
            Either "classification" or "regression" for type of model.
        n_classes: int (default 2)
            Number of classes to predict (only used in classification mode)
        batch_size: int (default 100)
            Batch size used by this model for training.
        """
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        if not isinstance(n_atom_feat, Sequence):
            n_atom_feat = [n_atom_feat] * n_weave
        if not isinstance(n_pair_feat, Sequence):
            n_pair_feat = [n_pair_feat] * n_weave
        n_layers = len(fully_connected_layer_sizes)
        if not isinstance(conv_weight_init_stddevs, Sequence):
            conv_weight_init_stddevs = [conv_weight_init_stddevs] * n_weave
        if not isinstance(weight_init_stddevs, Sequence):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, Sequence):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, Sequence):
            dropouts = [dropouts] * n_layers
        if not isinstance(activation_fns, Sequence):
            activation_fns = [activation_fns] * n_layers

        self.n_layers = n_layers
        self.n_weave = n_weave
        self.n_tasks = n_tasks
        self.n_atom_feat = n_atom_feat
        self.n_pair_feat = n_pair_feat
        self.n_hidden = n_hidden
        self.n_graph_feat = n_graph_feat
        self.mode = mode
        self.n_classes = n_classes
        self.weave_kernel_initializer = weave_kernel_initializer
        self.fully_connected_kernel_initializer = fully_connected_kernel_initializer
        self.batch_normalize = batch_normalize
        self.batch_normalize_kwargs = batch_normalize_kwargs
        self.fully_connected_layer_sizes = fully_connected_layer_sizes
        self.conv_weight_init_stddevs = conv_weight_init_stddevs
        self.weight_init_stddevs = weight_init_stddevs
        self.bias_init_consts = bias_init_consts
        self.dropouts = dropouts
        self.activation_fns = activation_fns
        self.final_conv_activation_fn = final_conv_activation_fn
        self.final_conv_kernel_size = final_conv_kernel_size
        self.gaussian_expand = gaussian_expand
        self.compress_post_gaussian_expansion = compress_post_gaussian_expansion
        self.batch_size = batch_size
        self.max_n_atoms = max_n_atoms
        self.max_n_pairs = max_n_pairs

        if weight_decay_penalty != 0.0:
            if weight_decay_penalty_type == 'l1':
                self.regularizer = keras.regularizers.l1(weight_decay_penalty)
            else:
                self.regularizer = keras.regularizers.l2(weight_decay_penalty)
        else:
            self.regularizer = None

    def build(self):
        atom_features = Input(shape=(self.max_n_atoms, self.n_atom_feat[0]))
        pair_features = Input(shape=(self.max_n_pairs, self.n_pair_feat[0]))
        # pair_split = Input(shape=tuple(), dtype=tf.int32)
        # atom_split = Input(shape=tuple(), dtype=tf.int32)
        atom_to_pair = Input(shape=(2, self.max_n_pairs), dtype=tf.int32)
        inputs = [atom_features, pair_features, atom_to_pair]
        for idx in range(self.n_weave):
            n_atom = inputs[0].shape[2]
            n_pair = inputs[1].shape[2]
            if idx < self.n_weave - 1:
                n_atom_next = self.n_atom_feat[idx + 1]
                n_pair_next = self.n_pair_feat[idx + 1]
            else:
                n_atom_next = self.n_hidden
                n_pair_next = self.n_hidden
            weave_layer_ind_A, weave_layer_ind_P = WeaveLayer(
                n_atom_input_feat=n_atom,
                n_pair_input_feat=n_pair,
                n_atom_output_feat=n_atom_next,
                n_pair_output_feat=n_pair_next,
                max_n_atoms=self.max_n_atoms,
                max_n_pairs=self.max_n_pairs,
                init=self.weave_kernel_initializer,
                batch_normalize=self.batch_normalize,
                batch_normalize_kwargs=self.batch_normalize_kwargs)(inputs)
            inputs = [weave_layer_ind_A, weave_layer_ind_P, atom_to_pair]
        # Final atom-layer convolution. Note this differs slightly from the paper
        # since we use a tanh activation as default. This seems necessary for numerical
        # stability.
        
        dense1 = Dense(self.n_graph_feat,
                    activation=self.final_conv_activation_fn)(weave_layer_ind_A)
        if self.batch_normalize:
            dense1 = BatchNormalization(**self.batch_normalize_kwargs)(dense1)
        dense1 = Conv1D(self.n_graph_feat, self.final_conv_kernel_size, activation=self.final_conv_activation_fn)(dense1)
        weave_gather = WeaveGather(self.batch_size,
                                   n_input=self.n_graph_feat)(dense1)

        if self.n_layers > 0:
            # Now fully connected layers
            input_layer = weave_gather
            for layer_size, weight_stddev, bias_const, dropout, activation_fn in zip(
                    self.fully_connected_layer_sizes, self.weight_init_stddevs,
                    self.bias_init_consts, self.dropouts, self.activation_fns):
                layer = Dense(
                    layer_size,
                    kernel_initializer=self.fully_connected_kernel_initializer,
                    bias_initializer=tf.constant_initializer(value=bias_const),
                    kernel_regularizer=self.regularizer)(input_layer)
                if dropout > 0.0:
                    layer = Dropout(rate=dropout)(layer)
                if self.batch_normalize:
                    # Should this allow for training?
                    layer = BatchNormalization(**self.batch_normalize_kwargs)(layer)
                layer = activation_fn(layer)
                input_layer = layer
            output = input_layer
        else:
            output = weave_gather

        n_tasks = self.n_tasks 
        if self.mode == 'classification':
            n_classes = self.n_classes
            logits = Reshape(
                (n_tasks, n_classes))(Dense(n_tasks * n_classes)(output))
            output = Softmax()(logits)
            outputs = [output, logits]
        else:
            output = Flatten()(output)
            output = Dense(n_tasks)(output)
            outputs = output
        model = tf.keras.Model(
            inputs=[atom_features, pair_features, atom_to_pair], 
            outputs=outputs
        )
        model.summary()
        return model