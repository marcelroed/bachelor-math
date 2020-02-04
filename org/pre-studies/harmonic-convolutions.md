# Notes about harmonic convolutions

## MNIST Call flow

* Args parsed
* Run main
* Placeholders for `x`, `y`, `lr`, `train_phase`
* `pred = deep_mnist(x, train_phase)`

### `deep_mnist`
##### Setup
This is a model for the MNIST-ROT dataset

Uses several parameters

* `nf, n2, nf3 = n_filters * filter_gain * i` - these are the number of filters used after each layer. This number increases by `filter_gain` for each layer.
* `n_classes` - the number of output classes
* `batch_size`
* `filter_size` - the size of each filter (in pixels?)
* `sm` - unused, called standard multiplication
* `n_rings` - Number of rings

##### Layers
Every operation imported from `hn_lite`
3 blocks of the same form
1. `mean_pool` - (not in 1st layer)
2. `conv2d` - harmonic convolution
3. `non_linearity` - takes in a 1d nonlinearity
4. `conv2d`
5. `batch_norm`

Final layer
1. `conv2d`
2. `sum_magnitudes`
3. `tf.reduce_mean`
4. `bias_add`


### `harmonic_network_lite`
#### `conv2d`
    ksize = size of square filter
    xsh = shape of x (batch_size, height, width, order, complex, channels)
    shape = [ksize, ksize, x_channels, out_channels]
    Q = get_weights_dict() -- Weights for each rotation order
    W - get_filters(W) -- Single frequency DFT on each ring for a polar-resampled patch

#### `get_filters`
    weights = get_interpolation_weights
    DFT = dft(N)[m, :]
    LPF = np.dot(DFT, weights)

#### `get_interpolation_weights`
