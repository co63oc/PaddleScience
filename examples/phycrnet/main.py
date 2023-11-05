# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PhyCRNet for solving spatiotemporal PDEs
Reference: https://github.com/isds-neu/PhyCRNet/
"""
import functions
import paddle
import scipy.io as scio

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(5)
    # set output directory
    OUTPUT_DIR = "./output_PhyCRNet" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    # set training hyper-parameters
    EPOCHS = 2000 if not args.epochs else args.epochs

    # set initial states for convlstm
    num_convlstm = 1
    (h0, c0) = (paddle.randn((1, 128, 16, 16)), paddle.randn((1, 128, 16, 16)))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))

    global num_time_batch
    global uv, dt, dx
    # grid parameters
    time_steps = 1001
    dt = 0.002
    dx = 1.0 / 128

    time_batch_size = 1000
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    model = ppsci.arch.PhyCRNet(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=dt,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps,
        effective_step=effective_step,
    )

    def _transform_out(_in, _out):
        return functions.transform_out(_in, _out, model)

    model.register_input_transform(functions.transform_in)
    model.register_output_transform(_transform_out)

    # use burgers_data.py to generate data
    data_file = "./output/burgers_1501x2x128x128.mat"
    data = scio.loadmat(data_file)
    uv = data["uv"]  # [t,c,h,w]

    # initial condition
    uv0 = uv[0:1, ...]
    input = paddle.to_tensor(uv0, dtype=paddle.get_default_dtype())

    initial_state = paddle.to_tensor(initial_state)
    dataset_obj = functions.Dataset(initial_state, input)
    (
        input_dict_train,
        label_dict_train,
        input_dict_val,
        label_dict_val,
    ) = dataset_obj.get(200)

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_train,
                "label": label_dict_train,
            },
        },
        ppsci.loss.FunctionalLoss(functions.train_loss_func),
        {
            "loss": lambda out: out["loss"],
        },
        name="sup_train",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
        },
        ppsci.loss.FunctionalLoss(functions.val_loss_func),
        {
            "loss": lambda out: out["loss"],
        },
        metric={"metric": ppsci.metric.FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # initialize solver
    ITERS_PER_EPOCH = 1
    scheduler = ppsci.optimizer.lr_scheduler.Step(
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        step_size=100,
        gamma=0.97,
        learning_rate=1e-4,
    )()
    optimizer = ppsci.optimizer.Adam(scheduler)(model)
    solver = ppsci.solver.Solver(
        model,
        constraint_pde,
        OUTPUT_DIR,
        optimizer,
        scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=50,
        validator=validator_pde,
        eval_with_no_grad=True,
    )

    # Used to set whether the graph is generated
    graph = False

    if not graph:
        # train model
        solver.train()
        # evaluate after finished training
        model.register_output_transform(functions.tranform_output_val)
        solver.eval()

        # save the model
        layer_state_dict = model.state_dict()
        paddle.save(layer_state_dict, "output/phycrnet.pdparams")
    else:
        import os

        fig_save_path = "output/figures/"
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path, True)
        layer_state_dict = paddle.load("output/phycrnet.pdparams")
        model.set_state_dict(layer_state_dict)
        model.register_output_transform(None)
        functions.output_graph(model, input_dict_val, fig_save_path, time_steps)
