import torch
import torch.utils.data.dataloader
import numpy as np
from collections import defaultdict
import numpy as np

import helpers


def train_value_model(value_model: helpers.BaseValueModel, device, optimizer, experience_replay: helpers.ExperienceReplay, batch_size: int, dataset_drop_ratio: float):
    value_model.train()
    helpers.optimizer_to(optimizer, device)

    dataset = []
    for action, env, reward in experience_replay.yield_training_tuples():
        for dataset_row in env.get_rotated_encoded_states_with_symmetry__value_model():
            if dataset_drop_ratio > 1e-5 and np.random.rand() < dataset_drop_ratio:
                continue
            dataset_row.append(reward)
            dataset.append(dataset_row)
    print("Value model dataset length", len(dataset))

    dataloader = helpers.torch_create_dataloader(dataset, device, batch_size=batch_size, shuffle=True, drop_last=True)

    predictions = defaultdict(list)

    for batch_input in dataloader:
        # for inp_tensor in batch_input: print(f"[train_value_model] inp_tensor {inp_tensor.shape} {inp_tensor.dtype}, {inp_tensor.device}")

        inputs = batch_input[:-1]
        actual_values = batch_input[-1].view((-1, 1)).float()

        pred_state_value = value_model.forward(*inputs)

        loss = torch.nn.functional.mse_loss(pred_state_value, actual_values)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        actual_values = actual_values.detach().cpu().numpy()
        pred_state_value = pred_state_value.detach().cpu().numpy()
        for i in range(len(inputs)):
            actual = float(actual_values[i, 0])
            pred = float(pred_state_value[i, 0])
            predictions[actual].append(pred)

        helpers.TENSORBOARD.append_scalar('value_loss', float(loss))