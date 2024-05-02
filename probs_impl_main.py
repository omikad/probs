import torch
import torch.utils.data.dataloader

import helpers
import probs_impl_common
import probs_impl_self_play
import probs_impl_train_q


class ProbsAlgorithmImpl:
    def __init__(self, params: probs_impl_common.Parameters, experience_replay: helpers.ExperienceReplay):
        self.params = params
        self.experience_replay = experience_replay

    def train_value_model(self, value_model: helpers.BaseValueModel, optimizer):
        experience_replay = self.experience_replay
        batch_size = self.params.value_batch_size

        value_model.train()

        dataset = []
        for action, env, reward in experience_replay.yield_training_tuples():
            dataset_row = env.get_rotated_encoded_state()
            dataset_row.append(reward)
            dataset.append(dataset_row)
        helpers.TENSORBOARD.append_scalar('value_data_len', len(dataset))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for batch_input in dataloader:
            inputs = batch_input[:-1]

            actual_values = batch_input[-1].view((-1, 1)).float()

            pred_state_value = value_model.forward(*inputs)

            loss = torch.nn.functional.mse_loss(pred_state_value, actual_values)

            helpers.TENSORBOARD.append_scalar('value_loss', loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


    def go_train(self, model_keeper: helpers.ModelKeeper):
        value_model = model_keeper.models['value']
        value_optimizer = model_keeper.optimizers['value']
        self_learning_model = model_keeper.models['self_learner']
        self_learning_optimizer = model_keeper.optimizers['self_learner']

        params = self.params
        experience_replay = self.experience_replay

        for high_level_i in range(params.n_high_level_iterations):
            experience_replay.clear()

            probs_impl_self_play.go_self_play(self_learning_model, params, experience_replay)

            print(f"High level iteration {high_level_i}/{params.n_high_level_iterations}")
            self.experience_replay.print_stats()

            self.train_value_model(value_model, value_optimizer)

            probs_impl_train_q.go_train_self_learning_model(
                experience_replay,
                self_learning_model,
                params,
                value_model,
                self_learning_optimizer
            )

            self_learning_model.eval()

            model_keeper.save_checkpoint(params.checkpoints_dir, "  ")

            agent = probs_impl_common.SelfLearningAgent(self_learning_model.__class__.__name__, model_keeper)
            agent.report_model_performance(
                env=params.create_env_func(),
                enemy_agent=params.evaluate_agent,
                n_evaluate_games=params.evaluate_n_games,
                n_max_steps=params.n_max_episode_steps,
                save_to_tf=True)
