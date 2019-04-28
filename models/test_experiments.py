import numpy as np
import matplotlib.pyplot as plt
from utils.algebra import quaternion_error, unit_quat, imu_integration, exp_mapping
from utils.visualization import Dynamic3DTrajectory
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

    def run_experiment(self, experiment_func, datasets_and_options):

        experiment_options = datasets_and_options["options"]
        datasets_and_options.pop("options")

        if "dynamic_plot" not in experiment_options.keys():
            experiment_options["dynamic_plot"] = False
        if "sparsing_factor" not in experiment_options.keys():
            experiment_options["sparsing_factor"] = 0

        # Request and save dataset if needed
        for dataset_key_tags in datasets_and_options.keys():
            if dataset_key_tags not in self.datasets.keys():
                self.datasets[dataset_key_tags] = self.dataset_loader_func(dataset_key_tags)

        # Run experiment list
        if experiment_func in self.available_experiments.keys():
            self.available_experiments[experiment_func](
                datasets=[self.datasets[exp_dataset_tag] for exp_dataset_tag in datasets_and_options.keys()],
                dataset_options=[datasets_and_options[key] for key in datasets_and_options.keys()],
                experiment_options=experiment_options,
                experiment_name=experiment_func)
        else:
            raise ValueError("This experiment is not available")

    def plot_predictions(self, datasets, dataset_options, experiment_options, experiment_name):

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
                    gt = np.append(dataset[1][:, :6], exp_mapping(dataset[1][:, 6:9]), axis=1)

        fig = self.plot_prediction(ground_truth=gt,
                                   model_prediction=predictions,
                                   comparative_prediction=comparisons,
                                   dynamic_plot=experiment_options["dynamic_plot"],
                                   sparsing_factor=experiment_options["sparsing_factor"])
        self.experiment_plot(fig, experiment_options, experiment_name=experiment_name)

    def training_progression(self, datasets, dataset_options, experiment_options, experiment_name):

        gt = []
        predictions = []
        comparisons = []

        j = 0
        next_model_num = 0
        while next_model_num != -1:
            figs = []
            model, next_model_num = self.model_loader(next_model_num)
            for i, dataset in enumerate(datasets):
                for option in dataset_options[i]:
                    if option == "predict":
                        predictions = model.predict(dataset, verbose=1)
                    elif option == "compare_prediction":
                        comparisons = self.alternative_prediction_method(np.squeeze(dataset[0]), self.window_len)
                    elif option == "ground_truth":
                        gt = np.append(dataset[1][:, :6], exp_mapping(dataset[1][:, 6:9]), axis=1)

            figs.append(self.plot_prediction(ground_truth=gt,
                                             model_prediction=predictions,
                                             comparative_prediction=comparisons),
                                             dynamic_plot=experiment_options["dynamic_plot"],
                                             sparsing_factor=experiment_options["sparsing_factor"])
            j += 1
            self.experiment_plot(figs[0], experiment_options, experiment_name=experiment_name, iteration=str(j))

    def iterate_model_output(self, datasets, dataset_options, experiment_options, experiment_name):

        gt = []
        predictions = []
        comparisons = []

        max_n_predictions = min([len(datasets[i][0]) for i in range(len(datasets))])
        n_predictions = None
        if "iterations" in experiment_options.keys():
            n_predictions = experiment_options["iterations"]
            assert n_predictions * self.window_len - 1 < max_n_predictions, \
                "The maximum number of iterations are {0} for the current window length of {1}".format(
                    int(np.floor(max_n_predictions / self.window_len)), self.window_len)

        for i, dataset in enumerate(datasets):

            if n_predictions is None:
                n_predictions = int(np.floor(len(dataset[0]) / self.window_len)) - 1

            for option in dataset_options[i]:

                if option == "predict":
                    model = self.model_loader()
                    model_predictions = np.zeros((n_predictions + 1, 10))
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
                    model_predictions = np.zeros((n_predictions + 1, 10))
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
                    gt = np.append(dataset[1][:n_predictions * self.window_len, :6],
                                   exp_mapping(dataset[1][:n_predictions * self.window_len, 6:9]), axis=1)

        predictions_x_axis = np.arange(0, n_predictions + 1) * self.window_len
        predictions_x_axis[1:] -= 1
        fig = self.plot_prediction(ground_truth=gt,
                                   model_prediction=predictions,
                                   comparative_prediction=comparisons,
                                   gt_x=np.arange(0, n_predictions * self.window_len),
                                   model_x=predictions_x_axis,
                                   comp_x=predictions_x_axis,
                                   dynamic_plot=experiment_options["dynamic_plot"],
                                   sparsing_factor=experiment_options["sparsing_factor"])
        self.experiment_plot(fig, experiment_options, experiment_name=experiment_name)

    @staticmethod
    def experiment_plot(figures, experiment_general_options, experiment_name, iteration=''):
        if experiment_general_options["output"] == "save":
            if isinstance(figures, (tuple, list)):
                for i, fig_i in enumerate(figures):
                    if "append_save_name" in experiment_general_options.keys():
                        fig_i.savefig('figures/fig_{0}_experiment_{1}{3}_{2}'.format(
                            experiment_name, experiment_general_options["append_save_name"], i, '_' + iteration))
                    else:
                        fig_i.savefig('figures/fig_{0}_experiment{2}_{1}'.format(experiment_name, i, '_' + iteration))
                    plt.close(fig_i)
            else:
                if "append_save_name" in experiment_general_options.keys():
                    figures.savefig('figures/fig_{0}_experiment_{1}{2}'.format(
                            experiment_name, experiment_general_options["append_save_name"], '_' + iteration))
                else:
                    figures.savefig('figures/fig_{0}_experiment{1}'.format(experiment_name, '_' + iteration))
                plt.close(figures)

        elif experiment_general_options["output"] == "show":
            plt.show()

    @staticmethod
    def plot_prediction(ground_truth, model_prediction, comparative_prediction, gt_x=None, model_x=None, comp_x=None,
                        dynamic_plot=False, sparsing_factor=0):

        if gt_x is None:
            gt_x = range(len(ground_truth))
        if model_x is None:
            model_x = range(len(model_prediction))
        if comp_x is None:
            comp_x = range(len(comparative_prediction))

        if dynamic_plot:
            dynamic_plot = Dynamic3DTrajectory([ground_truth[model_x, 0:3], model_prediction[:, 0:3]], sparsing_factor)
            dynamic_fig = dynamic_plot()
            plt.close(dynamic_fig)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(3, 1, 1)
        ax2 = fig1.add_subplot(3, 1, 2)
        ax3 = fig1.add_subplot(3, 1, 3)

        if isinstance(comparative_prediction, np.ndarray):
            ax1.plot(comp_x, comparative_prediction[:, 0], 'k')
            ax2.plot(comp_x, comparative_prediction[:, 1], 'k')
            ax3.plot(comp_x, comparative_prediction[:, 2], 'k')

        ax1.plot(gt_x, ground_truth[:, 0], 'b')
        ax1.plot(model_x, model_prediction[:, 0], 'r')
        ax2.plot(gt_x, ground_truth[:, 1], 'b')
        ax2.plot(model_x, model_prediction[:, 1], 'r')
        ax3.plot(gt_x, ground_truth[:, 2], 'b')
        ax3.plot(model_x, model_prediction[:, 2], 'r')
        ax1.set_title('pos_x')
        ax2.set_title('pos_y')
        ax3.set_title('pos_z')
        fig1.suptitle('Position predictions')

        fig2 = plt.figure()
        ax1 = fig2.add_subplot(3, 1, 1)
        ax2 = fig2.add_subplot(3, 1, 2)
        ax3 = fig2.add_subplot(3, 1, 3)

        if isinstance(comparative_prediction, np.ndarray):
            ax1.plot(comp_x, comparative_prediction[:, 3], 'k')
            ax2.plot(comp_x, comparative_prediction[:, 4], 'k')
            ax3.plot(comp_x, comparative_prediction[:, 5], 'k')

        ax1.plot(gt_x, ground_truth[:, 3], 'b')
        ax1.plot(model_x, model_prediction[:, 3], 'r')
        ax2.plot(gt_x, ground_truth[:, 4], 'b')
        ax2.plot(model_x, model_prediction[:, 4], 'r')
        ax3.plot(gt_x, ground_truth[:, 5], 'b')
        ax3.plot(model_x, model_prediction[:, 5], 'r')
        ax1.set_title('vel_x')
        ax2.set_title('vel_y')
        ax3.set_title('vel_z')
        fig2.suptitle('Velocity predictions')

        fig3 = plt.figure()
        ax1 = fig3.add_subplot(4, 1, 1)
        ax2 = fig3.add_subplot(4, 1, 2)
        ax3 = fig3.add_subplot(4, 1, 3)
        ax4 = fig3.add_subplot(4, 1, 4)

        gt_q = np.array([unit_quat(ground_truth[i, 6:10]).elements for i in gt_x])
        pred_q = np.array([unit_quat(model_prediction[i, 6:10]).elements for i in range(len(model_x))])

        if isinstance(comparative_prediction, np.ndarray):
            comp_pred_q = np.array([unit_quat(comparative_prediction[i, 6:10]).elements for i in range(len(comp_x))])
            ax1.plot(comp_x, comp_pred_q[:, 0], 'k')
            ax2.plot(comp_x, comp_pred_q[:, 1], 'k')
            ax3.plot(comp_x, comp_pred_q[:, 2], 'k')
            ax4.plot(comp_x, comp_pred_q[:, 3], 'k')

        ax1.plot(gt_x, gt_q[:, 0], 'b')
        ax1.plot(model_x, pred_q[:, 0], 'xkcd:orange')
        ax2.plot(gt_x, gt_q[:, 1], 'b')
        ax2.plot(model_x, pred_q[:, 1], 'xkcd:orange')
        ax3.plot(gt_x, gt_q[:, 2], 'b')
        ax3.plot(model_x, pred_q[:, 2], 'xkcd:orange')
        ax4.plot(gt_x, gt_q[:, 3], 'b')
        ax4.plot(model_x, pred_q[:, 3], 'xkcd:orange')
        ax1.set_title('att_w')
        ax2.set_title('att_x')
        ax3.set_title('att_y')
        ax4.set_title('att_z')
        fig3.suptitle('Attitude predictions')

        fig4 = plt.figure()
        ax1 = fig4.add_subplot(3, 1, 1)
        ax2 = fig4.add_subplot(3, 1, 2)
        ax3 = fig4.add_subplot(3, 1, 3)

        q_pred_e = [np.sin(quaternion_error(ground_truth[i, 6:10], model_prediction[i, 6:10]).angle)
                    for i in range(len(model_x))]

        if isinstance(comparative_prediction, np.ndarray):
            q_comp_pred_e = [np.sin(quaternion_error(ground_truth[i, 6:10], comparative_prediction[i, 6:10]).angle)
                             for i in range(len(comp_x))]
            ax1.plot(comp_x, np.linalg.norm(ground_truth[comp_x, :3] - comparative_prediction[:, :3], axis=1), 'k')
            ax2.plot(comp_x, np.linalg.norm(ground_truth[comp_x, 3:6] - comparative_prediction[:, 3:6], axis=1), 'k')
            ax3.plot(comp_x, q_comp_pred_e, 'k')

        ax1.plot(model_x, np.linalg.norm(ground_truth[model_x, :3] - model_prediction[:, :3], axis=1), 'r')
        ax1.set_title('position norm error')
        ax2.plot(model_x, np.linalg.norm(ground_truth[model_x, 3:6] - model_prediction[:, 3:6], axis=1), 'r')
        ax2.set_title('velocity norm error')
        ax3.plot(model_x, q_pred_e, 'xkcd:orange')
        ax3.set_title('attitude norm error')
        fig4.suptitle('Prediction vs manual integration errors')

        return fig1, fig2, fig3, fig4
