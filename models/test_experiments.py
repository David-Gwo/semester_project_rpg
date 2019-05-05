import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.algebra import imu_integration, log_mapping
from utils.visualization import Dynamic3DTrajectory
from tensorflow.python.keras.utils import Progbar
from mpl_toolkits.mplot3d import axes3d


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

        self.valid_plot_types = ["10-dof-state", "pre_integration"]

    def run_experiment(self, experiment_func, datasets_and_options):

        experiment_options = datasets_and_options["options"]
        datasets_and_options.pop("options")

        # Sanity checks of inputs
        if "plot_data" in experiment_options:
            plot_data = experiment_options["plot_data"]
            for key in plot_data.keys():
                if "type" not in plot_data[key].keys():
                    raise ValueError("The type of the data must be specified")
                elif plot_data[key]["type"] not in self.valid_plot_types:
                    raise ValueError("The type of the data must be one of the valid types: {0}".format(
                        self.valid_plot_types))
                if "dynamic_plot" not in plot_data[key].keys():
                    plot_data[key]["dynamic_plot"] = False
                if "sparsing_factor" not in plot_data[key].keys():
                    plot_data[key]["sparsing_factor"] = 0

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

        gt = {k: [] for k in experiment_options["plot_data"].keys()}
        predictions = {k: [] for k in experiment_options["plot_data"].keys()}
        comparisons = {k: [] for k in experiment_options["plot_data"].keys()}

        for i, dataset in enumerate(datasets):
            for option in dataset_options[i]:
                if option == "predict":
                    model = self.model_loader()
                    predictions = model.predict(dataset[0], verbose=1)
                    predictions = {out.name.split('/')[0]: predictions[i] for i, out in enumerate(model.outputs)}
                    predictions = {k: predictions[k] for k in experiment_options["plot_data"].keys()}
                elif option == "compare_prediction":
                    comparisons = self.alternative_prediction_method(np.squeeze(dataset[0]), self.window_len)
                elif option == "ground_truth":
                    gt = {k: dataset[1][k] for k in experiment_options["plot_data"].keys()}

        fig = self.draw_predictions(ground_truth=gt,
                                    model_prediction=predictions,
                                    comparative_prediction=comparisons,
                                    plot_options=experiment_options["plot_data"])
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
                        gt = dataset[1][:, :9]

            figs.append(self.draw_predictions(ground_truth=gt,
                                              model_prediction=predictions,
                                              comparative_prediction=comparisons,
                                              dynamic_plot=experiment_options["dynamic_plot"],
                                              sparsing_factor=experiment_options["sparsing_factor"]))
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
                    gt = dataset[1][:n_predictions * self.window_len, :9]

        predictions_x_axis = np.arange(0, n_predictions + 1) * self.window_len
        predictions_x_axis[1:] -= 1
        fig = self.draw_predictions(ground_truth=gt,
                                    model_prediction=predictions,
                                    comparative_prediction=comparisons,
                                    gt_x=np.arange(0, n_predictions * self.window_len),
                                    model_x=predictions_x_axis,
                                    comp_x=predictions_x_axis)
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

    def draw_predictions(self, ground_truth, model_prediction, comparative_prediction, plot_options,
                         gt_x=None, model_x=None, comp_x=None):

        for key in plot_options.keys():
            if plot_options[key]["type"] == "10-dof-state":
                return self.draw_state_output(ground_truth[key], model_prediction[key], comparative_prediction[key],
                                              plot_options[key], gt_x, model_x, comp_x)
            if plot_options[key]["type"] == "pre_integration":
                return self.draw_pre_integration(ground_truth[key], model_prediction[key], plot_options[key], gt_x,
                                                 model_x)

    def draw_state_output(self, ground_truth, model_prediction, comparative_prediction, options, gt_x, model_x, comp_x):
        if gt_x is None:
            gt_x = range(len(ground_truth))
        if model_x is None:
            model_x = range(len(model_prediction))
        if comp_x is None:
            comp_x = range(len(comparative_prediction))

        figs = []

        if options["dynamic_plot"]:
            if len(ground_truth) != len(model_prediction):
                tiled_prediction = np.zeros(ground_truth[:, 0:3].shape)
                for i in np.arange(len(model_prediction) - 1, 0, -1) - 1:
                    tiled_prediction[i * self.window_len:(i + 1) * self.window_len, :] = model_prediction[i, 0:3]
            else:
                tiled_prediction = model_prediction[:, 0:3]

            dynamic_plot = Dynamic3DTrajectory([ground_truth[:, 0:3], tiled_prediction], options["sparsing_factor"])
            _ = dynamic_plot()

        fig3d = plt.figure()
        ax = axes3d.Axes3D(fig3d)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], '-', color='b')
        ax.plot(model_prediction[:, 0], model_prediction[:, 1], model_prediction[:, 2], '-', color='r')

        figs.append(fig3d)

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
        ax1 = fig3.add_subplot(3, 1, 1)
        ax2 = fig3.add_subplot(3, 1, 2)
        ax3 = fig3.add_subplot(3, 1, 3)

        if isinstance(comparative_prediction, np.ndarray):
            comp_pred_q = log_mapping([comparative_prediction[i, 6:10] for i in range(len(comp_x))])
            ax1.plot(comp_x, comp_pred_q[:, 0], 'k')
            ax2.plot(comp_x, comp_pred_q[:, 1], 'k')
            ax3.plot(comp_x, comp_pred_q[:, 2], 'k')

            q_comp_pred_e = np.linalg.norm(ground_truth[model_x, 6:9] - comp_pred_q, axis=1)

        else:
            q_comp_pred_e = None

        ax1.plot(gt_x, ground_truth[:, 6], 'b')
        ax1.plot(model_x, model_prediction[:, 6], 'xkcd:orange')
        ax2.plot(gt_x, ground_truth[:, 7], 'b')
        ax2.plot(model_x, model_prediction[:, 7], 'xkcd:orange')
        ax3.plot(gt_x, ground_truth[:, 8], 'b')
        ax3.plot(model_x, model_prediction[:, 8], 'xkcd:orange')
        ax1.set_title('lie x')
        ax2.set_title('lie y')
        ax3.set_title('lie z')
        fig3.suptitle('Attitude predictions')

        fig4 = plt.figure()
        ax1 = fig4.add_subplot(3, 1, 1)
        ax2 = fig4.add_subplot(3, 1, 2)
        ax3 = fig4.add_subplot(3, 1, 3)

        if isinstance(comparative_prediction, np.ndarray):
            ax1.plot(comp_x, np.linalg.norm(ground_truth[comp_x, :3] - comparative_prediction[:, :3], axis=1), 'k')
            ax2.plot(comp_x, np.linalg.norm(ground_truth[comp_x, 3:6] - comparative_prediction[:, 3:6], axis=1), 'k')
            ax3.plot(comp_x, q_comp_pred_e, 'k')

        ax1.plot(model_x, np.linalg.norm(ground_truth[model_x, :3] - model_prediction[:, :3], axis=1), 'r')
        ax1.set_title('position norm error')
        ax2.plot(model_x, np.linalg.norm(ground_truth[model_x, 3:6] - model_prediction[:, 3:6], axis=1), 'r')
        ax2.set_title('velocity norm error')
        ax3.plot(model_x, np.linalg.norm(ground_truth[model_x, 6:9] - model_prediction[:, 6:9], axis=1), 'xkcd:orange')
        ax3.set_title('attitude norm error')
        fig4.suptitle('Prediction vs manual integration errors')

        figs.append([fig1, fig2, fig3, fig4])
        return tuple(figs)

    def draw_pre_integration(self, ground_truth, model_prediction, options, gt_x, model_x):

        gt_shape = ground_truth.shape

        if gt_x is None:
            gt_x = range(gt_shape[0])
        if model_x is None:
            model_x = range(len(model_prediction))

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(3, 1, 1)
        ax2 = fig1.add_subplot(3, 1, 2)
        ax3 = fig1.add_subplot(3, 1, 3)

        im1 = ax1.imshow(ground_truth[:, :, 0].T, cmap=cm.get_cmap('jet'), aspect='auto')
        im2 = ax2.imshow(ground_truth[:, :, 1].T, cmap=cm.get_cmap('jet'), aspect='auto')
        im3 = ax3.imshow(ground_truth[:, :, 2].T, cmap=cm.get_cmap('jet'), aspect='auto')

        fig1.colorbar(im1, ax=ax1)
        fig1.colorbar(im2, ax=ax2)
        fig1.colorbar(im3, ax=ax3)
        ax1.axes.get_xaxis().set_ticks([])
        ax2.axes.get_xaxis().set_ticks([])
        ax1.set_ylabel("x")
        ax2.set_ylabel("y")
        ax3.set_ylabel("z")
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        fig1.suptitle('Pre-integrated position increment')

        return fig1
