import numpy as np
import matplotlib.pyplot as plt
from utils import quaternion_error, unit_quat, imu_integration
from tensorflow.python.keras.utils import Progbar


class ExperimentManager:

    def __init__(self, window_len, final_epoch, model_loader_func, dataset_loader_func):
        self.window_len = window_len
        self.last_epoch_number = final_epoch
        self.model_loader = model_loader_func
        self.dataset_loader_func = dataset_loader_func
        self.alternative_prediction_method = imu_integration
        self.datasets = {}

        self.available_experiments = {
            "plot_predictions": self.plot_predictions,
            "training_progression": self.training_progression,
            "iterate_model_output": self.iterate_model_output
        }

    @staticmethod
    def plot_prediction(ground_truth, model_prediction, comparative_prediction, gt_x=None, model_x=None, comp_x=None):

        if gt_x is None:
            gt_x = range(len(ground_truth))
        if model_x is None:
            model_x = range(len(model_prediction))
        if comp_x is None:
            comp_x = range(len(comparative_prediction))

        if isinstance(comparative_prediction, np.ndarray):
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(3, 1, 1)
            ax2 = fig1.add_subplot(3, 1, 2)
            ax3 = fig1.add_subplot(3, 1, 3)

            ax1.plot(gt_x, ground_truth[:, 0], 'b')
            ax1.plot(model_x, model_prediction[:, 0], 'r')
            ax1.plot(comp_x, comparative_prediction[:, 0], 'k')
            ax2.plot(gt_x, ground_truth[:, 1], 'b')
            ax2.plot(model_x, model_prediction[:, 1], 'r')
            ax2.plot(comp_x, comparative_prediction[:, 1], 'k')
            ax3.plot(gt_x, ground_truth[:, 2], 'b')
            ax3.plot(model_x, model_prediction[:, 2], 'r')
            ax3.plot(comp_x, comparative_prediction[:, 2], 'k')
            ax1.set_title('pos_x')
            ax2.set_title('pos_y')
            ax3.set_title('pos_z')
            fig1.suptitle('Position predictions')

            fig2 = plt.figure()
            ax1 = fig2.add_subplot(3, 1, 1)
            ax2 = fig2.add_subplot(3, 1, 2)
            ax3 = fig2.add_subplot(3, 1, 3)

            ax1.plot(gt_x, ground_truth[:, 3], 'b')
            ax1.plot(model_x, model_prediction[:, 3], 'r')
            ax1.plot(comp_x, comparative_prediction[:, 3], 'k')
            ax2.plot(gt_x, ground_truth[:, 4], 'b')
            ax2.plot(model_x, model_prediction[:, 4], 'r')
            ax2.plot(comp_x, comparative_prediction[:, 4], 'k')
            ax3.plot(gt_x, ground_truth[:, 5], 'b')
            ax3.plot(model_x, model_prediction[:, 5], 'r')
            ax3.plot(comp_x, comparative_prediction[:, 5], 'k')
            ax1.set_title('vel_x')
            ax2.set_title('vel_y')
            ax3.set_title('vel_z')
            fig2.suptitle('Velocity predictions')

            fig3 = plt.figure()
            ax1 = fig3.add_subplot(4, 1, 1)
            ax2 = fig3.add_subplot(4, 1, 2)
            ax3 = fig3.add_subplot(4, 1, 3)
            ax4 = fig3.add_subplot(4, 1, 4)

            gt_q = np.array([unit_quat(ground_truth[i, 6:]).elements for i in gt_x])
            pred_q = np.array([unit_quat(model_prediction[i, 6:]).elements for i in range(len(model_x))])
            comp_pred_q = np.array([unit_quat(comparative_prediction[i, 6:]).elements for i in range(len(comp_x))])

            ax1.plot(gt_x, gt_q[:, 0], 'b')
            ax1.plot(model_x, pred_q[:, 0], 'r')
            ax1.plot(comp_x, comp_pred_q[:, 0], 'k')
            ax2.plot(gt_x, gt_q[:, 1], 'b')
            ax2.plot(model_x, pred_q[:, 1], 'r')
            ax2.plot(comp_x, comp_pred_q[:, 1], 'k')
            ax3.plot(gt_x, gt_q[:, 2], 'b')
            ax3.plot(model_x, pred_q[:, 2], 'r')
            ax3.plot(comp_x, comp_pred_q[:, 2], 'k')
            ax4.plot(gt_x, gt_q[:, 3], 'b')
            ax4.plot(model_x, pred_q[:, 3], 'r')
            ax4.plot(comp_x, comp_pred_q[:, 3], 'k')
            ax1.set_title('att_w')
            ax2.set_title('att_x')
            ax3.set_title('att_y')
            ax4.set_title('att_z')
            fig3.suptitle('Attitude predictions')

            q_pred_e = [np.sin(quaternion_error(ground_truth[i, 6:], model_prediction[i, 6:]).angle)
                        for i in range(len(model_x))]
            q_comp_pred_e = [np.sin(quaternion_error(ground_truth[i, 6:], comparative_prediction[i, 6:]).angle)
                             for i in range(len(comp_x))]

            fig4 = plt.figure()
            ax1 = fig4.add_subplot(3, 1, 1)
            ax1.plot(model_x, np.linalg.norm(ground_truth[model_x, :3] - model_prediction[:, :3], axis=1), 'r')
            ax1.plot(comp_x, np.linalg.norm(ground_truth[comp_x, :3] - comparative_prediction[:, :3], axis=1), 'k')
            ax1.set_title('position norm error')
            ax2 = fig4.add_subplot(3, 1, 2)
            ax2.plot(model_x, np.linalg.norm(ground_truth[model_x, 3:6] - model_prediction[:, 3:6], axis=1), 'r')
            ax2.plot(comp_x, np.linalg.norm(ground_truth[comp_x, 3:6] - comparative_prediction[:, 3:6], axis=1), 'k')
            ax2.set_title('velocity norm error')
            ax3 = fig4.add_subplot(3, 1, 3)
            ax3.plot(model_x, q_pred_e, 'r')
            ax3.plot(comp_x, q_comp_pred_e, 'k')
            ax3.set_title('attitude norm error')
            fig4.suptitle('Prediction vs manual integration errors')

            return fig1, fig2, fig3, fig4

        else:
            fig = plt.figure()
            ax = fig.add_subplot(3, 1, 1)
            ax.plot(gt_x, ground_truth[:, 0:3], 'b')
            ax.plot(model_x, model_prediction[:, 0:3], 'r')
            ax.set_title('position')
            ax = fig.add_subplot(3, 1, 2)
            ax.plot(gt_x, ground_truth[:, 3:6], 'b')
            ax.plot(model_x, model_prediction[:, 3:6], 'r')
            ax.set_title('velocity')
            ax = fig.add_subplot(3, 1, 3)
            ax.plot(gt_x, ground_truth[:, 6:10], 'b')
            ax.plot(model_x, model_prediction[:, 6:10], 'r')
            ax.set_title('attitude (quat)')

            return fig

    def run_experiment(self, experiment_func, datasets_and_options):

        experiment_options = datasets_and_options["options"]
        datasets_and_options.pop("options")

        # Request and save dataset if needed
        for dataset_key_tags in datasets_and_options.keys():
            if dataset_key_tags not in self.datasets.keys():
                self.datasets[dataset_key_tags] = self.dataset_loader_func(dataset_key_tags)

        # Run experiment list
        if experiment_func in self.available_experiments.keys():
            self.available_experiments[experiment_func](
                datasets=[self.datasets[exp_dataset_tag] for exp_dataset_tag in datasets_and_options.keys()],
                dataset_options=[datasets_and_options[key] for key in datasets_and_options.keys()],
                experiment_options=experiment_options)
        else:
            raise ValueError("This experiment is not available")

    def plot_predictions(self, datasets, dataset_options, experiment_options):

        gt = []
        predictions = []
        comparisons = []

        for i, dataset in enumerate(datasets):
            for option in dataset_options[i]:
                if option == "predict":
                    predictions = self.model_loader().predict(dataset, verbose=1)
                elif option == "compare_prediction":
                    comparisons = self.alternative_prediction_method(np.squeeze(dataset[0]), self.window_len)
                elif option == "ground_truth":
                    gt = dataset[1]

        fig = self.plot_prediction(ground_truth=gt, model_prediction=predictions, comparative_prediction=comparisons)
        self.experiment_plot(fig, experiment_options)

    def training_progression(self, datasets, dataset_options, experiment_options):

        gt = []
        predictions = []
        comparisons = []

        figs = []

        next_model_num = 0
        while next_model_num != -1:
            model, next_model_num = self.model_loader(next_model_num)
            for i, dataset in enumerate(datasets):
                for option in dataset_options[i]:
                    if option == "predict":
                        predictions = model.predict(dataset, verbose=1)
                    elif option == "compare_prediction":
                        comparisons = self.alternative_prediction_method(np.squeeze(dataset[0]), self.window_len)
                    elif option == "ground_truth":
                        gt = dataset[1]

            figs.append(self.plot_prediction(ground_truth=gt,
                                             model_prediction=predictions,
                                             comparative_prediction=comparisons))

        self.experiment_plot(figs, experiment_options)

    def iterate_model_output(self, datasets, dataset_options, experiment_options):

        gt = []
        predictions = []
        comparisons = []

        for i, dataset in enumerate(datasets):
            for option in dataset_options[i]:

                if option == "predict":
                    model = self.model_loader()
                    n_predictions = int(np.floor(len(dataset[0])/self.window_len)) - 1
                    model_predictions = np.zeros((n_predictions + 1, np.shape(dataset[1])[1]))
                    model_in = np.expand_dims(dataset[0][0, :, :, :], axis=0)
                    model_predictions[0, :] = model_in[0, self.window_len:, 0, 0]
                    progress_bar = Progbar(n_predictions)

                    for it in range(n_predictions):
                        progress_bar.update(it)
                        model_out = model.predict(model_in, verbose=0)
                        model_predictions[it, :] = model_out
                        model_in = np.expand_dims(dataset[0][self.window_len * (it + 1), :, :, :], axis=0)
                        model_in[0, self.window_len:, 0, :] = np.transpose(model_out)

                    model_predictions[-1, :] = model.predict(model_in, verbose=0)
                    progress_bar.update(n_predictions)
                    predictions = model_predictions

                elif option == "compare_prediction":
                    n_predictions = int(np.floor(len(dataset[0])/self.window_len)) - 1
                    model_predictions = np.zeros((n_predictions + 1, np.shape(dataset[1])[1]))
                    model_in = np.expand_dims(dataset[0][0, :, :, 0], axis=0)
                    model_predictions[0, :] = model_in[0, self.window_len:, 0]
                    progress_bar = Progbar(n_predictions)

                    for it in range(n_predictions):
                        progress_bar.update(it)
                        model_out = self.alternative_prediction_method(model_in, self.window_len, track_progress=False)
                        model_predictions[it, :] = model_out
                        model_in = np.expand_dims(dataset[0][self.window_len * (it + 1), :, :, 0], axis=0)
                        model_in[0, self.window_len:, 0] = np.squeeze(np.transpose(model_out))

                    model_predictions[-1, :] = self.alternative_prediction_method(model_in, self.window_len, track_progress=False)
                    progress_bar.update(n_predictions)
                    comparisons = model_predictions

                elif option == "ground_truth":
                    gt = dataset[1][self.window_len-1:, :]

        predictions_x_axis = np.arange(0, len(gt), self.window_len)
        fig = self.plot_prediction(ground_truth=gt,
                                   model_prediction=predictions,
                                   comparative_prediction=comparisons,
                                   model_x=predictions_x_axis,
                                   comp_x=predictions_x_axis)
        self.experiment_plot(fig, experiment_options)

    @staticmethod
    def experiment_plot(figures, experiment_general_options):
        if experiment_general_options["output"] == "save":
            if isinstance(figures, (tuple, list)) and len(figures) > 1:
                for i, fig_i in enumerate(figures):
                    if "append_save_name" in experiment_general_options.keys():
                        fig_i.savefig('figures/fig_{0}_{1}_{2}'.format(
                            "plot_predictions_experiment", experiment_general_options["append_save_name"], i))
                    else:
                        fig_i.savefig('figures/fig_{0}_{1}'.format("plot_predictions_experiment", i))
                    plt.close(fig_i)
            else:
                if "append_save_name" in experiment_general_options.keys():
                    figures.savefig(
                        'figures/fig_{0}_{1}'.format(
                            "plot_predictions_experiment", experiment_general_options["append_save_name"]))
                else:
                    figures.savefig('figures/fig_{0}'.format("plot_predictions_experiment"))
                plt.close(figures)

        elif experiment_general_options["output"] == "show":
            plt.show()
