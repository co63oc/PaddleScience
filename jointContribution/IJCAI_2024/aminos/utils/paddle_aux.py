# This file is generated by PaConvert ToolKit, please Don't edit it!
import paddle


def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


setattr(paddle.Tensor, "reshape", reshape)


def view(self, *args, **kwargs):
    if args:
        if len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                return paddle.reshape(self, args[0])  # To change reshape => view
            elif isinstance(args[0], str):
                return paddle.view(self, args[0])
            else:
                return paddle.reshape(self, list(args))  # To change reshape => view
        else:
            return paddle.reshape(self, list(args))  # To change reshape => view
    elif kwargs:
        key = [k for k in kwargs.keys()]
        if "dtype" in kwargs:
            return paddle.view(self, shape_or_dtype=kwargs[key[0]])
        else:
            return paddle.reshape(
                self, shape=kwargs[key[0]]
            )  # To change reshape => view


setattr(paddle.Tensor, "view", view)


def min(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.minimum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                return ret
        else:
            ret = paddle.min(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret


def max(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.maximum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                return ret
            return out_v
        else:
            ret = paddle.max(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret


def add(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        if alpha != 1:
            if not isinstance(y, paddle.Tensor):
                y = paddle.to_tensor(alpha * y)
            else:
                y = alpha * y
    else:
        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y)

    return paddle.add(self, y)


setattr(paddle.Tensor, "add", add)


def _FUNCTIONAL_PAD(x, pad, mode="constant", value=0.0, data_format="NCHW"):
    if len(x.shape) * 2 == len(pad) and mode == "constant":
        pad = (
            paddle.to_tensor(pad, dtype="int32")
            .reshape((-1, 2))
            .flip([0])
            .flatten()
            .tolist()
        )
    return paddle.nn.functional.pad(x, pad, mode, value, data_format)


def repeat(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.tile(self, args[0])
        else:
            return paddle.tile(self, list(args))
    elif kwargs:
        assert "repeats" in kwargs
        return paddle.tile(self, repeat_times=kwargs["repeats"])


setattr(paddle.Tensor, "repeat", repeat)


def min_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(
                self, *args, **kwargs
            )
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret


def max_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(
                self, *args, **kwargs
            )
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "min", min_class_func)
setattr(paddle.Tensor, "max", max_class_func)


def scatter_paddle(src, index, dim, out=None, reduce=None):
    if reduce is None:
        raise ValueError("'reduce' must be 'add', 'mean' or 'mul'.")

    if out is None:
        out = paddle.zeros_like(src)

    if reduce == "add":
        out = paddle.scatter_(out, index, src, overwrite=False)
    elif reduce == "mean":
        count = paddle.zeros_like(out)
        count = paddle.scatter_(count, index, paddle.ones_like(src), overwrite=False)
        out = paddle.scatter_(out, index, src, overwrite=False)
        count = paddle.clip(count, min=1)
        out = out / count
    elif reduce == "mul":
        out = paddle.scatter_(out, index, src, overwrite=False, reduce="multiply")
    else:
        raise ValueError("'reduce' must be 'add', 'mean' or 'mul'.")

    return out


def scatter_softmax_paddle(x_scaled, index, axis, overwrite=False):
    x_softmax = paddle.nn.functional.softmax(x_scaled, axis=-1)
    out = paddle.scatter(x_softmax, index, overwrite=False)
    return out
