import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.algebra import imu_integration, log_mapping, exp_mapping, quaternion_error, correct_quaternion_flip
from utils.visualization import Dynamic3DTrajectory
from utils.models import create_predictions_dict
from utils.directories import safe_mkdir_recursive
from tensorflow.python.keras.utils import Progbar
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import ImageGrid


class ExperimentManager:

    def __init__(self, window_len, final_epoch, model_loader_func, dataset_loader_func):
        self.window_len = window_len
        self.last_epoch_number = final_epoch
        self.model_loader = model_loader_func
        self.dataset_loader_func = dataset_loader_func
        self.alt_prediction_algo = imu_integration
        self.datasets = {}

        self.available_experiments = {
            "plot_predictions": self.plot_predictions,
            "training_progression": self.training_progression,
            "iterate_model_output": self.iterate_model_output
        }

        self.valid_plot_types = ["10-dof-state", "9-dof-state-lie", "pre_integration", "scalar"]
        self.output_type_vars = {
            "10-dof-state": {
                "shape": (10, )
            },
            "9-dof-state-lie": {
                "shape": (9, )
            }
        }

    def run_experiment(self, experiment_func, datasets_and_options):

        experiment_options = datasets_and_options["options"]
        datasets_and_options.pop("options")

        # Sanity checks of inputs
        if "output" not in experiment_options:
            experiment_options["output"] = "save"
        if experiment_options["output"] == "save":
            safe_mkdir_recursive('figures', overwrite=True)
        if "plot_data" in experiment_options:
            plot_data = experiment_options["plot_data"]
            for key in plot_data.keys():
                if "type" not in plot_data[key].keys():
                    raise ValueError("The type of the data must be specified")
                elif plot_data[key]["type"] not in self.valid_plot_types:
                    raise ValueError("The type of the data must be one of the valid types: {0}. Got {1} instead".format(
                        self.valid_plot_types, plot_data[key]["type"]))
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
                    predictions = create_predictions_dict(predictions, model)
                    predictions = {k: predictions[k] for k in experiment_options["plot_data"].keys()}
                elif option == "compare_prediction":
                    comparisons["state_output"] = self.alt_prediction_algo(np.squeeze(dataset[0]["imu_input"]),
                                                                           np.squeeze(dataset[0]["state_input"]))
                elif option == "ground_truth":
                    gt = {k: dataset[1][k] for k in experiment_options["plot_data"].keys()}

        fig = self.draw_predictions(ground_truth=gt,
                                    model_prediction=predictions,
                                    comp_prediction=comparisons,
                                    plot_options=experiment_options["plot_data"])
        self.experiment_plot(fig, experiment_options, experiment_name=experiment_name)

    def training_progression(self, datasets, dataset_options, experiment_options, experiment_name):

        gt = {k: [] for k in experiment_options["plot_data"].keys()}
        predictions = {k: [] for k in experiment_options["plot_data"].keys()}
        comparisons = {k: [] for k in experiment_options["plot_data"].keys()}

        j = 0
        next_model_num = 0
        while next_model_num != -1:
            figs = []
            model, next_model_num = self.model_loader(next_model_num)
            for i, dataset in enumerate(datasets):
                for option in dataset_options[i]:
                    if option == "predict":
                        predictions = model.predict(dataset[0], verbose=1)
                        predictions = create_predictions_dict(predictions, model)
                        predictions = {k: predictions[k] for k in experiment_options["plot_data"].keys()}
                    elif option == "compare_prediction":
                        comparisons = self.alt_prediction_algo(np.squeeze(dataset[0]), self.window_len)
                    elif option == "ground_truth":
                        gt = {k: dataset[1][k] for k in experiment_options["plot_data"].keys()}

            figs.append(self.draw_predictions(ground_truth=gt,
                                              model_prediction=predictions,
                                              comp_prediction=comparisons,
                                              plot_options=experiment_options["plot_data"]))
            j += 1
            self.experiment_plot(figs[0], experiment_options, experiment_name=experiment_name, iteration=str(j))

    def iterate_model_output(self, datasets, dataset_options, experiment_options, experiment_name):

        gt = {k: [] for k in experiment_options["plot_data"].keys()}
        predictions = {k: [] for k in experiment_options["plot_data"].keys()}
        comparisons = {k: [] for k in experiment_options["plot_data"].keys()}

        predictions_x_axis = None
        comparisons_x_axis = None
        gt_x_axis = None

        max_n_predictions = min([len(datasets[i][0]["imu_input"]) for i in range(len(datasets))])
        n_pred = None
        if "iterations" in experiment_options.keys():
            n_pred = experiment_options["iterations"]
            assert n_pred * self.window_len - 1 < max_n_predictions, \
                "The maximum number of iterations are {0} for the current window length of {1}".format(
                    int(np.floor(max_n_predictions / self.window_len)), self.window_len)
        assert len(experiment_options["plot_data"].keys()) == 1, \
            "Currently this experiment only supports one output. Got {0} instead: {1}".format(
                len(experiment_options["plot_data"].keys()), experiment_options["plot_data"].keys())
        assert "state_in" in experiment_options.keys()
        assert "state_out" in experiment_options.keys()

        output_name = list(experiment_options["plot_data"].keys())[0]
        output_size = self.output_type_vars[experiment_options["plot_data"][output_name]["type"]]["shape"]

        for i, dataset in enumerate(datasets):
            d_len = len(dataset[0]["imu_input"])

            if n_pred is None:
                n_predictions_p = int(np.floor((d_len - self.window_len) / (self.window_len - 1)) + 2)
                n_predictions_c = int(np.floor((d_len - self.window_len) / self.window_len) + 2)
            else:
                n_predictions_p = n_pred
                n_predictions_c = n_pred

            state_out_name = experiment_options["state_out"]["name"]
            state_in_name = experiment_options["state_in"]["name"]

            for option in dataset_options[i]:
                if option == "predict":
                    predictions_x_axis = np.zeros(n_predictions_p, dtype=np.int)
                    model = self.model_loader()
                    model_predictions = np.zeros((n_predictions_p, ) + output_size)
                    progress_bar = Progbar(n_predictions_p - 1)
                    model_out = {}
                    ds_i = 0
                    model_predictions[0] = dataset[1][output_name][0]
                    for it in range(n_predictions_p - 1):
                        progress_bar.update(it+1)
                        model_in = {k: np.expand_dims(dataset[0][k][ds_i], axis=0) for k in dataset[0].keys()}
                        if it > 0:
                            past_pred = model_out[state_out_name]
                            if experiment_options["state_out"]["lie"]:
                                past_pred = np.concatenate((past_pred[:, :6], exp_mapping(past_pred[:, 6:])), axis=1)
                            model_in[state_in_name] = past_pred
                        model_out = model.predict(model_in, verbose=0)
                        model_out = create_predictions_dict(model_out, model)
                        model_predictions[it+1, :] = model_out[output_name]
                        ds_i += self.window_len - 1
                        predictions_x_axis[it+1] = int(ds_i)
                        predictions_x_axis = predictions_x_axis.astype(np.int)

                    predictions[output_name] = model_predictions

                elif option == "compare_prediction":
                    model_predictions = np.zeros((n_predictions_c, 10))
                    comparisons_x_axis = np.zeros(n_predictions_c, dtype=np.int)
                    progress_bar = Progbar(n_predictions_c)
                    state_in = np.expand_dims(dataset[0][state_in_name][0], axis=0)
                    model_predictions[0, :] = state_in
                    ds_i = 0
                    for it in range(n_predictions_c):
                        progress_bar.update(it + 1)
                        model_out = self.alt_prediction_algo(
                            np.squeeze(np.expand_dims(dataset[0]["imu_input"][ds_i], axis=0), axis=-1), state_in, False)
                        model_predictions[it, :] = model_out
                        state_in = model_out
                        comparisons_x_axis[it] = int(ds_i)
                        comparisons_x_axis = comparisons_x_axis.astype(np.int)
                        ds_i += self.window_len - (1 if it == 0 else 0)

                    comparisons[output_name] = model_predictions

                elif option == "ground_truth":
                    state_in = np.expand_dims(dataset[0][state_in_name][0], axis=0)
                    if experiment_options["state_out"]["lie"]:
                        state_in = np.concatenate((state_in[:, :6], log_mapping(state_in[:, 6:])), axis=1)
                    state_in = np.tile(state_in, (self.window_len-1, 1))
                    gt = {k: dataset[1][k] for k in experiment_options["plot_data"].keys()}
                    gt[state_out_name] = np.concatenate((state_in, gt[state_out_name]), axis=0)
                    gt_x_axis = np.arange(0, len(gt[state_out_name]))

        fig = self.draw_predictions(ground_truth=gt,
                                    model_prediction=predictions,
                                    comp_prediction=comparisons,
                                    plot_options=experiment_options["plot_data"],
                                    gt_x=gt_x_axis.astype(np.int),
                                    model_x=predictions_x_axis,
                                    comp_x=comparisons_x_axis)
        self.experiment_plot(fig, experiment_options, experiment_name=experiment_name)

    @staticmethod
    def experiment_plot(figures, experiment_general_options, experiment_name, iteration=''):
        if iteration != '':
            iteration = '_' + iteration

        if experiment_general_options["output"] == "save":
            if isinstance(figures, (tuple, list)):
                for i, fig_i in enumerate(figures):
                    output_dir = 'figures/{0}'.format(i)
                    safe_mkdir_recursive(output_dir)
                    if "append_save_name" in experiment_general_options.keys():
                        fig_i.savefig('{0}/fig_{1}_experiment_{2}{3}.svg'.format(
                            output_dir, experiment_name, experiment_general_options["append_save_name"], iteration))
                    else:
                        fig_i.savefig('{0}/fig_{1}_experiment{2}.svg'.format(output_dir, experiment_name, iteration))
                    plt.close(fig_i)
            else:
                if "append_save_name" in experiment_general_options.keys():
                    figures.savefig('figures/fig_{0}_experiment_{1}{2}.svg'.format(
                            experiment_name, experiment_general_options["append_save_name"], iteration))
                else:
                    figures.savefig('figures/fig_{0}_experiment{1}.svg'.format(experiment_name, iteration))
                plt.close(figures)

        elif experiment_general_options["output"] == "show":
            plt.show()

    def draw_predictions(self, ground_truth, model_prediction, comp_prediction, plot_options,
                         gt_x=None, model_x=None, comp_x=None):

        figs = []

        for key in plot_options.keys():
            if plot_options[key]["type"] == "10-dof-state":
                figs.append(self.draw_state_output(ground_truth[key], model_prediction[key], comp_prediction[key],
                                                   plot_options[key], gt_x, model_x, comp_x))
            elif plot_options[key]["type"] == "9-dof-state-lie":
                figs.append(self.draw_state_output(ground_truth[key], model_prediction[key], comp_prediction[key],
                                                   plot_options[key], gt_x, model_x, comp_x))
            elif plot_options[key]["type"] == "pre_integration":
                figs.append(self.draw_pre_integration(ground_truth[key], model_prediction[key], gt_x, model_x,
                                                      title=key))
            elif plot_options[key]["type"] == "scalar":
                figs.append(self.draw_scalar_comparison(ground_truth[key], model_prediction[key], comp_prediction[key],
                                                        gt_x, model_x, comp_x, plot_options[key]))

        return figs

    def draw_state_output(self, ground_truth, model_prediction, comparative_prediction, options, gt_x, model_x, comp_x):
        if gt_x is None:
            gt_x = range(len(ground_truth))
        if model_x is None:
            model_x = range(len(model_prediction))
        if comp_x is None:
            comp_x = range(len(comparative_prediction))

        figs = []
        comp_available = isinstance(comparative_prediction, np.ndarray)

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
        if comp_available:
            ax.plot(comparative_prediction[:, 0], comparative_prediction[:, 1], comparative_prediction[:, 2], '-',
                    color='k')
            ax.legend(['g_truth', 'prediction', 'integration'])
        else:
            ax.legend(['g_truth', 'prediction'])
        ax.set_xlabel('m')
        ax.set_ylabel('m')
        ax.set_zlabel('m')

        figs.append(fig3d)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(3, 1, 1)
        ax2 = fig1.add_subplot(3, 1, 2)
        ax3 = fig1.add_subplot(3, 1, 3)

        ax1.plot(gt_x, ground_truth[:, 0], 'b')
        ax1.plot(model_x, model_prediction[:, 0], 'r')
        ax2.plot(gt_x, ground_truth[:, 1], 'b')
        ax2.plot(model_x, model_prediction[:, 1], 'r')
        ax3.plot(gt_x, ground_truth[:, 2], 'b')
        ax3.plot(model_x, model_prediction[:, 2], 'r')
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_ylabel(r'$p_x\;[m]$')
        ax2.set_ylabel(r'$p_y\;[m]$')
        ax3.set_ylabel(r'$p_z\;[m]$')
        ax3.set_xlabel('sample #')

        if comp_available:
            ax1.plot(comp_x, comparative_prediction[:, 0], 'k')
            ax2.plot(comp_x, comparative_prediction[:, 1], 'k')
            ax3.plot(comp_x, comparative_prediction[:, 2], 'k')
            ax1.legend(['g_truth', 'prediction', 'integration'], loc='upper right')
        else:
            ax1.legend(['g_truth', 'prediction'], loc='upper right')
        fig1.tight_layout()

        fig2 = plt.figure()
        ax1 = fig2.add_subplot(3, 1, 1)
        ax2 = fig2.add_subplot(3, 1, 2)
        ax3 = fig2.add_subplot(3, 1, 3)

        ax1.plot(gt_x, ground_truth[:, 3], 'b')
        ax1.plot(model_x, model_prediction[:, 3], 'r')
        ax2.plot(gt_x, ground_truth[:, 4], 'b')
        ax2.plot(model_x, model_prediction[:, 4], 'r')
        ax3.plot(gt_x, ground_truth[:, 5], 'b')
        ax3.plot(model_x, model_prediction[:, 5], 'r')
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_ylabel(r'$v_x\;[m/s]$')
        ax2.set_ylabel(r'$v_y\;[m/s]$')
        ax3.set_ylabel(r'$v_z\;[m/s]$')
        ax3.set_xlabel('sample #')

        if comp_available:
            ax1.plot(comp_x, comparative_prediction[:, 3], 'k')
            ax2.plot(comp_x, comparative_prediction[:, 4], 'k')
            ax3.plot(comp_x, comparative_prediction[:, 5], 'k')
            ax1.legend(['g_truth', 'prediction', 'integration'], loc='upper right')
        else:
            ax1.legend(['g_truth', 'prediction'], loc='upper right')
        fig2.tight_layout()

        q_pred_e = q_comp_pred_e = None
        fig3 = plt.figure()
        if options["type"] == "10-dof-state":
            ax1 = fig3.add_subplot(4, 1, 1)
            ax2 = fig3.add_subplot(4, 1, 2)
            ax3 = fig3.add_subplot(4, 1, 3)
            ax4 = fig3.add_subplot(4, 1, 4)

            model_prediction[:, 6:] = correct_quaternion_flip(model_prediction[:, 6:])

            ax1.plot(gt_x, ground_truth[:, 6], 'b')
            ax1.plot(model_x, model_prediction[:, 6], 'r')
            ax2.plot(gt_x, ground_truth[:, 7], 'b')
            ax2.plot(model_x, model_prediction[:, 7], 'r')
            ax3.plot(gt_x, ground_truth[:, 8], 'b')
            ax3.plot(model_x, model_prediction[:, 8], 'r')
            ax4.plot(gt_x, ground_truth[:, 9], 'b')
            ax4.plot(model_x, model_prediction[:, 9], 'r')
            ax1.set_ylabel(r'$q_w$')
            ax2.set_ylabel(r'$q_x$')
            ax3.set_ylabel(r'$q_y$')
            ax4.set_ylabel(r'$q_z$')
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax3.set_xticks([])
            ax4.set_xlabel('sample #')

            q_pred_e = [abs(np.sin(quaternion_error(ground_truth[i, 6:10], model_prediction[i, 6:10]).angle))
                        for i in range(len(model_x))]

            if comp_available:
                ax1.plot(comp_x, comparative_prediction[:, 6], 'k')
                ax2.plot(comp_x, comparative_prediction[:, 7], 'k')
                ax3.plot(comp_x, comparative_prediction[:, 8], 'k')
                ax4.plot(comp_x, comparative_prediction[:, 8], 'k')
                q_comp_pred_e = [abs(np.sin(quaternion_error(ground_truth[i, 6:10], comparative_prediction[i, 6:10]).angle))
                                 for i in range(len(comp_x))]
                ax1.legend(['g_truth', 'prediction', 'integration'], loc='upper right')
            else:
                q_comp_pred_e = None
                ax1.legend(['g_truth', 'prediction'], loc='upper right')

        elif options["type"] == "9-dof-state-lie":
            ax1 = fig3.add_subplot(3, 1, 1)
            ax2 = fig3.add_subplot(3, 1, 2)
            ax3 = fig3.add_subplot(3, 1, 3)

            ax1.plot(gt_x, ground_truth[:, 6], 'xkcd:aquamarine')
            ax1.plot(model_x, model_prediction[:, 6], 'xkcd:orange')
            ax2.plot(gt_x, ground_truth[:, 7], 'xkcd:aquamarine')
            ax2.plot(model_x, model_prediction[:, 7], 'xkcd:orange')
            ax3.plot(gt_x, ground_truth[:, 8], 'xkcd:aquamarine')
            ax3.plot(model_x, model_prediction[:, 8], 'xkcd:orange')
            ax1.set_ylabel(r'$\mathfrak{q}_x$')
            ax2.set_ylabel(r'$\mathfrak{q}_y$')
            ax3.set_ylabel(r'$\mathfrak{q}_z$')
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax3.set_xlabel('sample #')

            q_pred_e = np.linalg.norm(ground_truth[model_x, 6:9] - model_prediction[:, 6:9], axis=1)
            q_error = quaternion_error(exp_mapping(ground_truth[model_x, 6:9]), exp_mapping(model_prediction[:, 6:9]))
            q_pred_e = np.append(np.expand_dims(q_pred_e, axis=1),
                                 np.expand_dims(np.array([abs(np.sin(q_e.angle)) for q_e in q_error]), axis=1), axis=1)

            if comp_available:
                comp_pred_q = log_mapping([comparative_prediction[i, 6:10] for i in range(len(comp_x))])
                ax1.plot(comp_x, comp_pred_q[:, 0], 'xkcd:grey')
                ax2.plot(comp_x, comp_pred_q[:, 1], 'xkcd:grey')
                ax3.plot(comp_x, comp_pred_q[:, 2], 'xkcd:grey')
                q_comp_pred_e = np.linalg.norm(ground_truth[comp_x, 6:9] - comp_pred_q, axis=1)
                q_error = quaternion_error(exp_mapping(ground_truth[comp_x, 6:9]), comparative_prediction[:, 6:10])
                q_comp_pred_e = np.append(np.expand_dims(q_comp_pred_e, axis=1),
                                          np.expand_dims(np.array([abs(np.sin(q_e.angle)) for q_e in q_error]), axis=1),
                                          axis=1)

                ax1.legend(['g_truth', 'prediction', 'integration'], loc='upper right')
            else:
                q_comp_pred_e = None
                ax1.legend(['g_truth', 'prediction'], loc='upper right')
        fig3.tight_layout()

        fig4 = plt.figure()
        ax1 = fig4.add_subplot(3, 1, 1)
        ax2 = fig4.add_subplot(3, 1, 2)
        ax3 = fig4.add_subplot(3, 1, 3)
        ax3.set_xlabel('sample #')

        ax1.plot(model_x, np.linalg.norm(ground_truth[model_x, :3] - model_prediction[:, :3], axis=1), 'r')
        ax2.plot(model_x, np.linalg.norm(ground_truth[model_x, 3:6] - model_prediction[:, 3:6], axis=1), 'r')
        if options["type"] == "10-dof-state":
            ax3.plot(model_x, q_pred_e, 'r')
        elif options["type"] == "9-dof-state-lie":
            ax3.plot(model_x, q_pred_e[:, 1], 'r')
            ax3.plot(model_x, q_pred_e[:, 0], 'xkcd:orange')
        if comp_available:
            ax1.plot(comp_x, np.linalg.norm(ground_truth[comp_x, :3] - comparative_prediction[:, :3], axis=1), 'k')
            ax2.plot(comp_x, np.linalg.norm(ground_truth[comp_x, 3:6] - comparative_prediction[:, 3:6], axis=1), 'k')
            if options["type"] == "10-dof-state":
                ax3.plot(comp_x, q_comp_pred_e, 'k')
            elif options["type"] == "9-dof-state-lie":
                ax3.plot(comp_x, q_comp_pred_e[:, 1], 'k')
                ax3.plot(comp_x, q_comp_pred_e[:, 0], 'xkcd:grey')

            ax1.legend(['prediction', 'integration'], loc='upper right')
            ax3.legend(['prediction', 'pred. lie', 'integration', 'integ. lie'], loc='upper right')

        else:
            ax1.legend(['prediction'], loc='upper right')
            ax3.legend(['prediction', 'pred. lie'], loc='upper right')

        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_ylabel(r'$\|\|\mathbf{p}-\mathbf{\hat{p}}\|\|_2^2\;[m]$')
        ax2.set_ylabel(r'$\|\|\mathbf{v}-\mathbf{\hat{v}}\|\|_2^2\;[m/s]$')
        ax3.set_ylabel(r'$\|\sin\left(\left(\mathbf{q}^{-1}\mathbf{\hat{q}}\right)_\measuredangle \right)\|$'
                       if options["type"] == "10-dof-state" else
                       r'$\|\|\mathfrak{q}-\mathfrak{\hat{q}}\|\|_2^2$')

        fig4.tight_layout()

        figs.append([fig1, fig2, fig3, fig4])
        return tuple(figs)

    def draw_pre_integration(self, ground_truth, model_prediction, gt_x, model_x, title):

        fig1 = plt.figure()

        grid1 = ImageGrid(fig1, 311, nrows_ncols=(1, 3), axes_pad=0.15, share_all=True, cbar_location="right",
                          cbar_mode="single", cbar_size="7%", cbar_pad=0.15, aspect=False)
        ax1, ax4, ax7 = grid1.axes_all
        grid2 = ImageGrid(fig1, 312, nrows_ncols=(1, 3), axes_pad=0.15, share_all=True, cbar_location="right",
                          cbar_mode="single", cbar_size="7%", cbar_pad=0.15, aspect=False)
        ax2, ax5, ax8 = grid2.axes_all
        grid3 = ImageGrid(fig1, 313, nrows_ncols=(1, 3), axes_pad=0.15, share_all=True, cbar_location="right",
                          cbar_mode="single", cbar_size="7%", cbar_pad=0.15, aspect=False)
        ax3, ax6, ax9 = grid3.axes_all

        model_prediction = np.reshape(model_prediction, (len(model_prediction), self.window_len, -1))

        diff_1 = np.abs(ground_truth[:, :, 0].T - model_prediction[:, :, 0].T)
        diff_2 = np.abs(ground_truth[:, :, 1].T - model_prediction[:, :, 1].T)
        diff_3 = np.abs(ground_truth[:, :, 2].T - model_prediction[:, :, 2].T)

        vmin_1 = min([np.amin(ground_truth[:, :, 0]), np.amin(model_prediction[:, :, 0]), np.amin(diff_1)])
        vmax_1 = max([np.amax(ground_truth[:, :, 0]), np.amax(model_prediction[:, :, 0]), np.amax(diff_1)])
        vmin_2 = min([np.amin(ground_truth[:, :, 1]), np.amin(model_prediction[:, :, 1]), np.amin(diff_2)])
        vmax_2 = max([np.amax(ground_truth[:, :, 1]), np.amax(model_prediction[:, :, 1]), np.amax(diff_2)])
        vmin_3 = min([np.amin(ground_truth[:, :, 2]), np.amin(model_prediction[:, :, 2]), np.amin(diff_3)])
        vmax_3 = max([np.amax(ground_truth[:, :, 2]), np.amax(model_prediction[:, :, 2]), np.amax(diff_3)])

        ax1.imshow(ground_truth[:, :, 0].T, cmap=cm.get_cmap('jet'), vmin=vmin_1, vmax=vmax_1, aspect='auto')
        ax2.imshow(ground_truth[:, :, 1].T, cmap=cm.get_cmap('jet'), vmin=vmin_2, vmax=vmax_2, aspect='auto')
        ax3.imshow(ground_truth[:, :, 2].T, cmap=cm.get_cmap('jet'), vmin=vmin_3, vmax=vmax_3, aspect='auto')
        ax4.imshow(model_prediction[:, :, 0].T, cmap=cm.get_cmap('jet'), vmin=vmin_1, vmax=vmax_1, aspect='auto')
        ax5.imshow(model_prediction[:, :, 1].T, cmap=cm.get_cmap('jet'), vmin=vmin_2, vmax=vmax_2, aspect='auto')
        ax6.imshow(model_prediction[:, :, 2].T, cmap=cm.get_cmap('jet'), vmin=vmin_3, vmax=vmax_3, aspect='auto')
        im7 = ax7.imshow(diff_1, cmap=cm.get_cmap('jet'), vmin=vmin_1, vmax=vmax_1, aspect='auto')
        im8 = ax8.imshow(diff_2, cmap=cm.get_cmap('jet'), vmin=vmin_2, vmax=vmax_2, aspect='auto')
        im9 = ax9.imshow(diff_3, cmap=cm.get_cmap('jet'), vmin=vmin_3, vmax=vmax_3, aspect='auto')

        grid1.cbar_axes[0].colorbar(im7)
        grid2.cbar_axes[0].colorbar(im8)
        grid3.cbar_axes[0].colorbar(im9)

        ax1.axes.get_xaxis().set_ticks([])
        ax2.axes.get_xaxis().set_ticks([])
        ax4.axes.get_xaxis().set_ticks([])
        ax5.axes.get_xaxis().set_ticks([])

        ax1.set_ylabel("x")
        ax2.set_ylabel("y")
        ax3.set_ylabel("z")

        ax1.set_title("True")
        ax4.set_title("Predicted")
        ax7.set_title("Error")

        fig1.suptitle(title)

        return fig1

    def draw_scalar_comparison(self, ground_truth, model_prediction, comp_prediction, gt_x, model_x, comp_x, plot_op):

        fig1 = plt.figure()
        plt.plot(ground_truth, 'b')
        plt.plot(model_prediction, 'r')
        if isinstance(comp_prediction, np.ndarray):
            plt.plot(comp_prediction, 'k')
            plt.legend(["g_truth", "prediction", "integration"])
        else:
            plt.legend(["g_truth", "prediction"])
        plt.title(plot_op["title"])
        plt.xlabel('sample')
        plt.ylabel(plot_op["y_label"])
        return fig1
