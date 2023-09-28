import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.initializer as init

paddle.seed(seed=123)


def nn_parameter(input, requires_grad=False):
    param = paddle.create_parameter(
        shape=input.shape,
        dtype=input.numpy().dtype,
        default_initializer=paddle.nn.initializer.Assign(input),
    )
    param.stop_gradient = not requires_grad
    return param


def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(
            paddle.uniform(shape=tensor.shape, dtype=tensor.dtype, min=a, max=b)
        )
        return tensor


def uniform_(tensor: paddle.Tensor, a: float, b: float) -> paddle.Tensor:
    """Modify tensor inplace using uniform_.

    Args:
        tensor (paddle.Tensor): Paddle Tensor.
        a (float): min value.
        b (float): max value.

    Returns:
        paddle.Tensor: Initialized tensor.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> param = paddle.empty((128, 256), "float32")
        >>> param = ppsci.utils.initializer.uniform_(param, -1, 1)
    """
    return _no_grad_uniform_(tensor, a, b)


# torch default nonlinearity='leaky_relu'
def init_kaiming_normal_(param, mode="", nonlinearity="leaky_relu"):
    initializer = paddle.nn.initializer.KaimingNormal(nonlinearity=nonlinearity)
    initializer(param)


class USCNN(nn.Layer):
    def __init__(
        self, h, nx, ny, n_var_in=1, n_var_out=1, init_way="ortho", k=5, s=1, p=2
    ):
        super(USCNN, self).__init__()
        """
        Extract basic information
        """
        self.init_way = init_way
        self.n_var_in = n_var_in
        self.n_var_out = n_var_out
        self.k = k
        self.s = 1
        self.p = 2
        self.delta_x = h
        self.nx = nx
        self.ny = ny

        """
        Define net
        """
        self.relu = nn.ReLU()
        self.us = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode="bicubic")
        self.conv1 = nn.Conv2D(self.n_var_in, 16, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2D(16, 32, kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv2D(32, 16, kernel_size=k, stride=s, padding=p)
        self.conv4 = nn.Conv2D(16, self.n_var_out, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle = nn.PixelShuffle(1)
        if self.init_way is not None:
            self._initialize_weights()
        # Specify filter
        dx_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, -8.0, 0.0, 8.0, -1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdx = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdx.weight = nn_parameter(dx_filter, requires_grad=False)

        dy_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, -8.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 8.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdy = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdy.weight = nn_parameter(dy_filter, requires_grad=False)

        lap_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [-1.0, 16.0, -60.0, 16.0, -1.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
            / self.delta_x
        )
        self.convlap = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convlap.weight = nn_parameter(lap_filter, requires_grad=False)

    def forward(self, x):
        x = self.us(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        # x=(self.conv4(x))
        return x

    def _initialize_weights(self):
        if self.init_way == "kaiming":
            init_kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv3.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv4.weight)
        elif self.init_way == "ortho":
            init1 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init1(self.conv1.weight)
            init2 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init2(self.conv2.weight)
            init3 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init3(self.conv3.weight)
            init4 = init.Orthogonal()
            init4(self.conv4.weight)
        elif self.init_way == "uniform":
            self.apply(self.__init_weights)
        else:
            print("Only Kaiming or Orthogonal initializer can be used!")
            exit()

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            bound = 1 / np.sqrt(np.prod(m.weight.shape[1:]))
            uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                uniform_(m.bias, -bound, bound)


class USCNNSep(nn.Layer):
    def __init__(
        self, h, nx, ny, n_var_in=1, n_var_out=1, init_way=None, k=5, s=1, p=2
    ):
        super(USCNNSep, self).__init__()
        """
        Extract basic information
        """
        self.init_way = init_way
        self.n_var_in = n_var_in
        self.n_var_out = n_var_out
        self.k = k
        self.s = 1
        self.p = 2
        self.delta_x = h
        self.nx = nx
        self.ny = ny
        """
        Define net
        """
        W1 = 16
        W2 = 32
        self.relu = nn.ReLU()
        self.us = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode="bicubic")
        self.conv1 = nn.Conv2D(self.n_var_in, W1, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2D(W1, W2, kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv2D(W2, W1, kernel_size=k, stride=s, padding=p)
        self.conv4 = nn.Conv2D(W1, self.n_var_out, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle1 = nn.PixelShuffle(1)
        self.conv11 = nn.Conv2D(self.n_var_in, W1, kernel_size=k, stride=s, padding=p)
        self.conv22 = nn.Conv2D(W1, W2, kernel_size=k, stride=s, padding=p)
        self.conv33 = nn.Conv2D(W2, W1, kernel_size=k, stride=s, padding=p)
        self.conv44 = nn.Conv2D(W1, self.n_var_out, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle11 = nn.PixelShuffle(1)
        self.conv111 = nn.Conv2D(self.n_var_in, W1, kernel_size=k, stride=s, padding=p)
        self.conv222 = nn.Conv2D(W1, W2, kernel_size=k, stride=s, padding=p)
        self.conv333 = nn.Conv2D(W2, W1, kernel_size=k, stride=s, padding=p)
        self.conv444 = nn.Conv2D(W1, self.n_var_out, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle111 = nn.PixelShuffle(1)
        if self.init_way is not None:
            self._initialize_weights()
        # Specify filter
        dxi_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, -8.0, 0.0, 8.0, -1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdxi = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdxi = paddle.nn.Conv2D(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 5),
            stride=1,
            padding=0,
            bias_attr=None,
        )
        self.convdxi.weight = nn_parameter(dxi_filter, requires_grad=False)

        deta_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, -8.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 8.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdeta = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdeta.weight = nn_parameter(deta_filter, requires_grad=False)

        lap_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [-1.0, 16.0, -60.0, 16.0, -1.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
            / self.delta_x
        )
        self.convlap = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convlap.weight = nn_parameter(lap_filter, requires_grad=False)

    def forward(self, x):
        x = self.us(x)
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        x1 = self.pixel_shuffle1(self.conv4(x1))

        x2 = self.relu(self.conv11(x))
        x2 = self.relu(self.conv22(x2))
        x2 = self.relu(self.conv33(x2))
        x2 = self.pixel_shuffle11(self.conv44(x2))

        x3 = self.relu(self.conv111(x))
        x3 = self.relu(self.conv222(x3))
        x3 = self.relu(self.conv333(x3))
        x3 = self.pixel_shuffle111(self.conv444(x3))
        return paddle.concat([x1, x2, x3], axis=1)

    def _initialize_weights(self):
        if self.init_way == "kaiming":
            init_kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv3.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv4.weight)
            init_kaiming_normal_(
                self.conv11.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(
                self.conv22.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(
                self.conv33.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(self.conv44.weight)
            init_kaiming_normal_(
                self.conv111.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(
                self.conv222.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(
                self.conv333.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(self.conv444.weight)
        elif self.init_way == "ortho":
            init1 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init1(self.conv1.weight)
            init2 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init2(self.conv2.weight)
            init3 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init3(self.conv3.weight)
            init4 = init.Orthogonal()
            init4(self.conv4.weight)
            init11 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init11(self.conv11.weight)
            init22 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init22(self.conv22.weight)
            init33 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init33(self.conv33.weight)
            init44 = init.Orthogonal()
            init44(self.conv44.weight)
            init111 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init111(self.conv111.weight)
            init222 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init222(self.conv222.weight)
            init333 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init333(self.conv333.weight)
            init444 = init.Orthogonal()
            init444(self.conv444.weight)
        else:
            print("Only Kaiming or Orthogonal initializer can be used!")
            exit()


class USCNNSepPhi(nn.Layer):
    def __init__(
        self, output_size, n_var_in=1, n_var_out=1, init_way=None, k=5, s=1, p=2
    ):
        super(USCNNSepPhi, self).__init__()
        """
        Extract basic information
        """
        self.init_way = init_way
        self.n_var_in = n_var_in
        self.n_var_out = n_var_out
        self.k = k
        self.s = 1
        self.p = 2
        self.delta_x = 1 / output_size
        self.output_size = output_size

        """
        Define net
        """
        W1 = 16
        W2 = 32
        self.relu = nn.ReLU()
        self.us = nn.Upsample(size=[self.output_size, self.output_size], mode="bicubic")
        self.conv1 = nn.Conv2D(self.n_var_in, W1, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2D(W1, W2, kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv2D(W2, W1, kernel_size=k, stride=s, padding=p)
        self.conv4 = nn.Conv2D(W1, self.n_var_out, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle1 = nn.PixelShuffle(1)
        self.conv11 = nn.Conv2D(self.n_var_in, W1, kernel_size=k, stride=s, padding=p)
        self.conv22 = nn.Conv2D(W1, W2, kernel_size=k, stride=s, padding=p)
        self.conv33 = nn.Conv2D(W2, W1, kernel_size=k, stride=s, padding=p)
        self.conv44 = nn.Conv2D(W1, self.n_var_out, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle11 = nn.PixelShuffle(1)
        if self.init_way is not None:
            self._initialize_weights()
        # Specify filter
        shrink_filter = paddle.to_tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.conv_shrink = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.conv_shrink.weight = nn_parameter(shrink_filter, requires_grad=False)

        dx_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, -8.0, 0.0, 8.0, -1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdx = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdx.weight = nn_parameter(dx_filter, requires_grad=False)

        dy_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, -8.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 8.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdy = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdy.weight = nn_parameter(dy_filter, requires_grad=False)

        lap_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [-1.0, 16.0, -60.0, 16.0, -1.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
            / self.delta_x
        )
        self.convlap = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convlap.weight = nn_parameter(lap_filter, requires_grad=False)

    def forward(self, x):
        x = self.us(x)
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        x1 = self.pixel_shuffle1(self.conv4(x1))

        x2 = self.relu(self.conv11(x))
        x2 = self.relu(self.conv22(x2))
        x2 = self.relu(self.conv33(x2))
        x2 = self.pixel_shuffle11(self.conv44(x2))

        return paddle.concat([x1, x2], axis=1)

    def _initialize_weights(self):
        if self.init_way == "kaiming":
            init_kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv3.weight, mode="fan_out", nonlinearity="relu")
            init_kaiming_normal_(self.conv4.weight)
            init_kaiming_normal_(
                self.conv11.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(
                self.conv22.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(
                self.conv33.weight, mode="fan_out", nonlinearity="relu"
            )
            init_kaiming_normal_(self.conv44.weight)
        elif self.init_way == "ortho":
            init1 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init1(self.conv1.weight)
            init2 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init2(self.conv2.weight)
            init3 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init3(self.conv3.weight)
            init4 = init.Orthogonal()
            init4(self.conv4.weight)
            init11 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init11(self.conv11.weight)
            init22 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init22(self.conv22.weight)
            init33 = init.Orthogonal(gain=init.calculate_gain("relu"))
            init33(self.conv33.weight)
            init44 = init.Orthogonal()
            init44(self.conv44.weight)
        else:
            print("Only Kaiming or Orthogonal initializer can be used!")
            exit()


def flatchannel(output_size, n_var_in, n_var_out, W1, W2, k, s, p):
    return nn.Sequential(
        nn.Upsample(size=[int(output_size / 5), int(output_size / 5)], mode="bicubic"),
        nn.Conv2D(n_var_in, W1, kernel_size=k, stride=s, padding=p),
        nn.ReLU(),
        nn.Conv2D(W1, W2, kernel_size=k, stride=s, padding=p),
        nn.ReLU(),
        nn.Conv2D(W2, W1, kernel_size=k, stride=s, padding=p),
        nn.ReLU(),
        nn.Conv2D(W1, n_var_out, kernel_size=k, stride=s, padding=p),
        nn.PixelShuffle(1),
    )


class DDBasic(nn.Layer):
    """docstring for DDBasic"""

    def __init__(
        self, output_size, n_var_in=1, n_var_out=1, init_way=None, k=5, s=1, p=2
    ):
        super(DDBasic, self).__init__()
        self.init_way = init_way
        self.n_var_in = n_var_in
        self.n_var_out = n_var_out
        self.k = k
        self.s = 1
        self.p = 2
        self.delta_x = 1 / output_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.us = nn.Upsample(size=[self.output_size, self.output_size], mode="bicubic")
        self.DS = nn.Upsample(
            size=[int(self.output_size / 2), int(self.output_size / 2)], mode="bicubic"
        )

        # Define Forward
        for i in range(25):
            exec(
                "self.Phi_"
                + str(int(i))
                + "=flatchannel(self.output_size,self.n_var_in,self.n_var_out,W1,W2,self.k,self.s,self.p)"
            )
            exec(
                "self.P_"
                + str(int(i))
                + "=flatchannel(self.output_size,self.n_var_in,self.n_var_out,W1,W2,self.k,self.s,self.p)"
            )
        if self.init_way is not None:
            self._initialize_weights()
        shrink_filter = paddle.to_tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.conv_shrink = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.conv_shrink.weight = nn_parameter(shrink_filter, requires_grad=False)

        dx_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, -8.0, 0.0, 8.0, -1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdx = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdx.weight = nn_parameter(dx_filter, requires_grad=False)

        dy_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, -8.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 8.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdy = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdy.weight = nn_parameter(dy_filter, requires_grad=False)

        lap_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [-1.0, 16.0, -60.0, 16.0, -1.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
            / self.delta_x
        )
        self.convlap = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convlap.weight = nn_parameter(lap_filter, requires_grad=False)

    def forward(self, x):
        xup = self.us(x)
        x1 = paddle.zeros(xup[:, 0:1, :, :].shape)
        x2 = paddle.zeros(xup[:, 1:2, :, :].shape)
        for i in range(5):
            for j in range(5):
                exec(
                    "x1[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="
                    + "self.Phi_"
                    + str(int(i * 5 + j))
                    + "(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])"
                )
                exec(
                    "x2[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="
                    + "self.P_"
                    + str(int(i * 5 + j))
                    + "(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])"
                )

        """
        x1[:,0,0:int(self.output_size/2),0:int(self.output_size/2)]=self.Phi_1(x[:,:,0:int(self.output_size/2),0:int(self.output_size/2)])
        x1[:,0,0:int(self.output_size/2),int(self.output_size/2):self.output_size]=self.Phi_2(x[:,:,0:int(self.output_size/2),int(self.output_size/2):self.output_size])
        x1[:,0,int(self.output_size/2):self.output_size,0:int(self.output_size/2)]=self.Phi_3(x[:,:,int(self.output_size/2):self.output_size,0:int(self.output_size/2)])
        x1[:,0,int(self.output_size/2):self.output_size,int(self.output_size/2):self.output_size]=self.Phi_4(x[:,:,int(self.output_size/2):self.output_size,int(self.output_size/2):self.output_size])

        x2[:,0,0:int(self.output_size/2),0:int(self.output_size/2)]=self.P_1(x[:,:,0:int(self.output_size/2),0:int(self.output_size/2)])
        x2[:,0,0:int(self.output_size/2),int(self.output_size/2):self.output_size]=self.P_2(x[:,:,0:int(self.output_size/2),int(self.output_size/2):self.output_size])
        x2[:,0,int(self.output_size/2):self.output_size,0:int(self.output_size/2)]=self.P_3(x[:,:,int(self.output_size/2):self.output_size,0:int(self.output_size/2)])
        x2[:,0,int(self.output_size/2):self.output_size,int(self.output_size/2):self.output_size]=self.P_4(x[:,:,int(self.output_size/2):self.output_size,int(self.output_size/2):self.output_size])
        """
        return paddle.concat([x1, x2], axis=1)

    def _initialize_weights(self, m=None):
        if self.init_way == "kaiming":
            if isinstance(m, nn.Conv2D):
                init.KaimingNormal()(m.weight)
        elif self.init_way == "ortho":
            if isinstance(m, nn.Conv2D):
                init.KaimingNormal()(m.weight)
        else:
            print("Only Kaiming or Orthogonal initializer can be used!")
            exit()


class DDBasicSepNoPhi(nn.Layer):
    """docstring for DDBasic"""

    def __init__(
        self, output_size, n_var_in=1, n_var_out=1, init_way=None, k=5, s=1, p=2
    ):
        super(DDBasicSepNoPhi, self).__init__()
        self.init_way = init_way
        self.n_var_in = n_var_in
        self.n_var_out = n_var_out
        self.k = k
        self.s = 1
        self.p = 2
        self.delta_x = 1 / output_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.us = nn.Upsample(size=[self.output_size, self.output_size], mode="bicubic")
        self.DS = nn.Upsample(
            size=[int(self.output_size / 2), int(self.output_size / 2)], mode="bicubic"
        )

        # Define Forward
        for i in range(25):
            exec(
                "self.U_"
                + str(int(i))
                + "=flatchannel(self.output_size,self.n_var_in,self.n_var_out,W1,W2,self.k,self.s,self.p)"
            )
            exec(
                "self.V_"
                + str(int(i))
                + "=flatchannel(self.output_size,self.n_var_in,self.n_var_out,W1,W2,self.k,self.s,self.p)"
            )
            exec(
                "self.P_"
                + str(int(i))
                + "=flatchannel(self.output_size,self.n_var_in,self.n_var_out,W1,W2,self.k,self.s,self.p)"
            )
        if self.init_way is not None:
            self._initialize_weights()
        shrink_filter = paddle.to_tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.conv_shrink = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.conv_shrink.weight = nn_parameter(shrink_filter, requires_grad=False)

        dx_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, -8.0, 0.0, 8.0, -1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdx = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdx.weight = nn_parameter(dx_filter, requires_grad=False)

        dy_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, -8.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 8.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
        )
        self.convdy = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convdy.weight = nn_parameter(dy_filter, requires_grad=False)

        lap_filter = (
            paddle.to_tensor(
                [
                    [
                        [
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [-1.0, 16.0, -60.0, 16.0, -1.0],
                            [0.0, 0.0, 16.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 0.0],
                        ]
                    ]
                ]
            )
            / 12.0
            / self.delta_x
            / self.delta_x
        )
        self.convlap = nn.Conv2D(1, 1, (5, 5), stride=1, padding=0, bias_attr=None)
        self.convlap.weight = nn_parameter(lap_filter, requires_grad=False)

    def forward(self, x):
        xup = self.us(x)
        x1 = paddle.zeros(xup[:, 0:1, :, :].shape)
        x2 = paddle.zeros(xup[:, 0:1, :, :].shape)
        x3 = paddle.zeros(xup[:, 0:1, :, :].shape)
        for i in range(5):
            for j in range(5):
                # pdb.set_trace()
                exec(
                    "x1[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="
                    + "self.U_"
                    + str(int(i * 5 + j))
                    + "(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])"
                )
                exec(
                    "x2[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="
                    + "self.V_"
                    + str(int(i * 5 + j))
                    + "(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])"
                )
                exec(
                    "x3[:,0:1,i*l:(i+1)*l,j*l:(j+1)*l]="
                    + "self.P_"
                    + str(int(i * 5 + j))
                    + "(xup[:,:,i*l:(i+1)*l,j*l:(j+1)*l])"
                )

                # exec("x1[:,0,i*l:(i+1)*l,j*l:(j+1)*l]="+"self.Phi_"+str(int(i*5+j))+"(x)")
                # exec("x2[:,0,i*l:(i+1)*l,j*l:(j+1)*l]="+"self.P_"+str(int(i*5+j))+"(x)")
        """
        x1[:,0,0:int(self.output_size/2),0:int(self.output_size/2)]=self.Phi_1(x[:,:,0:int(self.output_size/2),0:int(self.output_size/2)])
        x1[:,0,0:int(self.output_size/2),int(self.output_size/2):self.output_size]=self.Phi_2(x[:,:,0:int(self.output_size/2),int(self.output_size/2):self.output_size])
        x1[:,0,int(self.output_size/2):self.output_size,0:int(self.output_size/2)]=self.Phi_3(x[:,:,int(self.output_size/2):self.output_size,0:int(self.output_size/2)])
        x1[:,0,int(self.output_size/2):self.output_size,int(self.output_size/2):self.output_size]=self.Phi_4(x[:,:,int(self.output_size/2):self.output_size,int(self.output_size/2):self.output_size])

        x2[:,0,0:int(self.output_size/2),0:int(self.output_size/2)]=self.P_1(x[:,:,0:int(self.output_size/2),0:int(self.output_size/2)])
        x2[:,0,0:int(self.output_size/2),int(self.output_size/2):self.output_size]=self.P_2(x[:,:,0:int(self.output_size/2),int(self.output_size/2):self.output_size])
        x2[:,0,int(self.output_size/2):self.output_size,0:int(self.output_size/2)]=self.P_3(x[:,:,int(self.output_size/2):self.output_size,0:int(self.output_size/2)])
        x2[:,0,int(self.output_size/2):self.output_size,int(self.output_size/2):self.output_size]=self.P_4(x[:,:,int(self.output_size/2):self.output_size,int(self.output_size/2):self.output_size])
        """
        return paddle.concat([x1, x2, x3], axis=1)

    def _initialize_weights(self, m=None):
        if self.init_way == "kaiming":
            if isinstance(m, nn.Conv2D):
                init.KaimingNormal()(m.weight)
        elif self.init_way == "ortho":
            if isinstance(m, nn.Conv2D):
                init.KaimingNormal()(m.weight)
        else:
            print("Only Kaiming or Orthogonal initializer can be used!")
            exit()
