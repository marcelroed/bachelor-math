# Notes from Group Equivariant Convolutional Networks (Welling, Cohen)

Introducing G-CNNs (Group equivariant Convolutional Neural Networks), generalization of CNNs that reduce sample complexity by exploiting symmetries. __G-convolutions__ are a new type of layer that has a substantially higher degree of weight sharing than regular convolution layers, increase expressive capacity of the network without increasing the number of parameters.

## 1. Introduction
Ordinary CNNs are _translation equivariant_ (shifting then applying = applying then shifting).

This paper shows how CNNs can be generalized to exploit larger groups of symmetries, including rotations and reflections.

First show that ordinary CNNs are translationally equivariant, but fail to equivary with more general transformations.

Show that G-convolutions, pooling, arbitrary pointwise nonlinearities, batch normalization and residual blocks are all equivariant, and therefore compatible with G-CNNs.

## 2. Structured & Equivariant Representations
Ordinary NNs produce progressively more abstract representations by mapping th einput through a series of parameterized functions, endowed with a minimal internal structure (linear space $\R^n$).

Instead, this paper constructs representations that have the structure of a linear $G$-space, for some chosen group $G$. Meaning that each vector in the representation space has a pose associated with it, which can be transformed by the elements of some group of transformations $G$. Additionally this allows us to model data more efficiently: ~A filter in a G-CNN detects co-occurrences of features that have the preferred relative pose, and can match such a feature constellation in every global pose through an operation called the G-convolution.~

A representation space can obtain its structure from other rep. spaces to which it is connected. For this we need each layer $\Phi$ mapping one representation to another to be _structure preserving_. For $G$-spaces this means that $Phi$ has to be equivariant:

$$\Phi(T_gx) = T_g'\Phi(x),$$

So transforming an input $x$ by $g$ ($T_gx$) then passing it through the learned map $\Phi$ should give the same result as first mapping $x$ through $\Phi$ and then transforming the representation. Note that $T$ and $T'$ need not be the same. We only need that for these operators any two transformations $g, h$ give $T(gh) = T(g)T(h)$ (so $T$ is a linear representation of $G$).
