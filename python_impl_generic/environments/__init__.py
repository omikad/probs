from . import connect_four_env
from . import connect_four_nn
from . import my_chess_env
from . import my_chess_nn
from . import reversi_env
from . import reversi_nn


__all__ = [
    "get_create_env_func",
    "create_value_model",
    "create_self_learning_model",
]


def get_create_env_func(ARGS, env_name: str):
    if env_name == "mychess4x5": return my_chess_env.create_env_45_func
    if env_name == "mychess6x6": return my_chess_env.create_env_66_func
    if env_name == "mychess8x6": return my_chess_env.create_env_86_func
    if env_name == "mychess8x8": return my_chess_env.create_env_88_func
    if env_name == "connect_four": return connect_four_env.create_env_func
    if env_name == "reversi": return reversi_env.create_env_func

    raise Exception(f"Unknown env {env_name}")


def create_value_model(env_name: str, value_model_class_name: str):
    if env_name.startswith("mychess"):
        return getattr(my_chess_nn, value_model_class_name)()
    if env_name.startswith("connect_four"):
        return getattr(connect_four_nn, value_model_class_name)()
    if env_name.startswith("reversi"):
        return getattr(reversi_nn, value_model_class_name)()
    raise Exception(f"Unknown model {value_model_class_name} for env {env_name}")


def create_self_learning_model(env_name: str, self_learning_model_class_name: str):
    if env_name.startswith("mychess"):
        return getattr(my_chess_nn, self_learning_model_class_name)()
    if env_name.startswith("connect_four"):
        return getattr(connect_four_nn, self_learning_model_class_name)()
    if env_name.startswith("reversi"):
        return getattr(reversi_nn, self_learning_model_class_name)()
    raise Exception(f"Unknown model {self_learning_model_class_name} for env {env_name}")


