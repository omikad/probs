from dataclasses import dataclass
from typing import Callable

from . import connect_four_env
from . import connect_four_nn
from . import reversi_env
from . import reversi_nn


__all__ = [
    "get_create_env_func",
    "create_value_model",
    "create_self_learning_model",
    "get_default_train_params",
]


@dataclass
class EnvRegistration:
    create_env_func: Callable
    create_value_model_func: Callable
    create_self_learning_model_func: Callable
    default_training_params: dict


ENVIRONMENTS = {
    "connect_four": EnvRegistration(
        create_env_func=connect_four_env.create_env_func,
        create_value_model_func=connect_four_nn.create_value_model,
        create_self_learning_model_func=connect_four_nn.create_self_learning_model,
        default_training_params=connect_four_nn.DEFAULT_TRAIN_PARAMS,
    ),
    "reversi": EnvRegistration(
        create_env_func=reversi_env.create_env_func,
        create_value_model_func=reversi_nn.create_value_model,
        create_self_learning_model_func=reversi_nn.create_self_learning_model,
        default_training_params=reversi_nn.DEFAULT_TRAIN_PARAMS,
    ),
}


def get_create_env_func(env_name: str):
    return ENVIRONMENTS[env_name].create_env_func


def create_value_model(ARGS, env_name: str, value_model_class_name: str):
    return ENVIRONMENTS[env_name].create_value_model_func(ARGS, value_model_class_name)


def create_self_learning_model(ARGS, env_name: str, self_learning_model_class_name: str):
    return ENVIRONMENTS[env_name].create_self_learning_model_func(ARGS, self_learning_model_class_name)


def get_default_train_params(ARGS, env_name: str):
    return ENVIRONMENTS[env_name].default_training_params