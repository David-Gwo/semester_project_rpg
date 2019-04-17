import numpy as np
import matplotlib.pyplot as plt
from utils import quaternion_error, unit_quat


class ExperimentManager:

    def __init__(self, final_epoch, model_loader_func, dataset_loader_func):
        self.deep_model = None
        self.last_epoch_number = final_epoch
        self.model_loader = model_loader_func
        self.dataset_loader_func = dataset_loader_func
        self.datasets = {}

    @staticmethod
    def plot_prediction(gt, prediction, manual_pred):

        if manual_pred is not 0 and manual_pred is not None:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(3, 1, 1)
            ax2 = fig1.add_subplot(3, 1, 2)
            ax3 = fig1.add_subplot(3, 1, 3)

            ax1.plot(gt[:, 0], 'b')
            ax1.plot(prediction[:, 0], 'r')
            ax1.plot(manual_pred[:, 0], 'k')
            ax2.plot(gt[:, 1], 'b')
            ax2.plot(prediction[:, 1], 'r')
            ax2.plot(manual_pred[:, 1], 'k')
            ax3.plot(gt[:, 2], 'b')
            ax3.plot(prediction[:, 2], 'r')
            ax3.plot(manual_pred[:, 2], 'k')
            ax1.set_title('pos_x')
            ax2.set_title('pos_y')
            ax3.set_title('pos_z')
            fig1.suptitle('Position predictions')

            fig2 = plt.figure()
            ax1 = fig2.add_subplot(3, 1, 1)
            ax2 = fig2.add_subplot(3, 1, 2)
            ax3 = fig2.add_subplot(3, 1, 3)

            ax1.plot(gt[:, 3], 'b')
            ax1.plot(prediction[:, 3], 'r')
            ax1.plot(manual_pred[:, 3], 'k')
            ax2.plot(gt[:, 4], 'b')
            ax2.plot(prediction[:, 4], 'r')
            ax2.plot(manual_pred[:, 4], 'k')
            ax3.plot(gt[:, 5], 'b')
            ax3.plot(prediction[:, 5], 'r')
            ax3.plot(manual_pred[:, 5], 'k')
            ax1.set_title('vel_x')
            ax2.set_title('vel_y')
            ax3.set_title('vel_z')
            fig2.suptitle('Velocity predictions')

            fig3 = plt.figure()
            ax1 = fig3.add_subplot(4, 1, 1)
            ax2 = fig3.add_subplot(4, 1, 2)
            ax3 = fig3.add_subplot(4, 1, 3)
            ax4 = fig3.add_subplot(4, 1, 4)

            gt_q = np.array([unit_quat(gt[i, 6:]).elements for i in range(len(gt))])
            pred_q = np.array([unit_quat(prediction[i, 6:]).elements for i in range(len(gt))])
            mpred_q = np.array([unit_quat(manual_pred[i, 6:]).elements for i in range(len(gt))])

            ax1.plot(gt_q[:, 0], 'b')
            ax1.plot(pred_q[:, 0], 'r')
            ax1.plot(mpred_q[:, 0], 'k')
            ax2.plot(gt_q[:, 1], 'b')
            ax2.plot(pred_q[:, 1], 'r')
            ax2.plot(mpred_q[:, 1], 'k')
            ax3.plot(gt_q[:, 2], 'b')
            ax3.plot(pred_q[:, 2], 'r')
            ax3.plot(mpred_q[:, 2], 'k')
            ax4.plot(gt_q[:, 3], 'b')
            ax4.plot(pred_q[:, 3], 'r')
            ax4.plot(mpred_q[:, 3], 'k')
            ax1.set_title('att_w')
            ax2.set_title('att_x')
            ax3.set_title('att_y')
            ax4.set_title('att_z')
            fig3.suptitle('Attitude predictions')

            q_pred_e = [np.sin(quaternion_error(gt[i, 6:], prediction[i, 6:]).angle) for i in range(len(gt))]
            q_mpred_e = [np.sin(quaternion_error(gt[i, 6:], manual_pred[i, 6:]).angle) for i in range(len(gt))]

            fig4 = plt.figure()
            ax1 = fig4.add_subplot(3, 1, 1)
            ax1.plot(np.linalg.norm(gt[:, :3] - prediction[:, :3], axis=1), 'r')
            ax1.plot(np.linalg.norm(gt[:, :3] - manual_pred[:, :3], axis=1), 'k')
            ax1.set_title('position norm error')
            ax2 = fig4.add_subplot(3, 1, 2)
            ax2.plot(np.linalg.norm(gt[:, 3:6] - prediction[:, 3:6], axis=1), 'r')
            ax2.plot(np.linalg.norm(gt[:, 3:6] - manual_pred[:, 3:6], axis=1), 'k')
            ax2.set_title('velocity norm error')
            ax3 = fig4.add_subplot(3, 1, 3)
            ax3.plot(q_pred_e, 'r')
            ax3.plot(q_mpred_e, 'k')
            ax3.set_title('attitude norm error')
            fig4.suptitle('Prediction vs manual integration errors')

            return fig1, fig2, fig3, fig4

        else:
            fig = plt.figure()
            ax = fig.add_subplot(3, 1, 1)
            ax.plot(gt[:, 0:3], 'b')
            ax.plot(prediction[:, 0:3], 'r')
            ax.set_title('position')
            ax = fig.add_subplot(3, 1, 2)
            ax.plot(gt[:, 3:6], 'b')
            ax.plot(prediction[:, 3:6], 'r')
            ax.set_title('velocity')
            ax = fig.add_subplot(3, 1, 3)
            ax.plot(gt[:, 6:10], 'b')
            ax.plot(prediction[:, 6:10], 'r')
            ax.set_title('attitude (quat)')

            return fig

    def run_experiment(self, experiment, exp_dataset_tags):

        # Request and save dataset if needed
        if exp_dataset_tags not in self.datasets.keys():
            self.datasets[exp_dataset_tags] = self.dataset_loader_func(exp_dataset_tags)

        if experiment == 'plot_predictions':
            self.plot_regression_predictions()

    # if self.config.generate_training_progression:
    #     model_pos = 0
    #     while model_pos != -1:
    #         model_pos = self.recover_model_from_checkpoint(mode="test", model_used_pos=model_pos)
    #         self.trained_model_dir = self.config.checkpoint_dir + self.model_version_number + '/'
    #         self.evaluate_model(save_figures=True)
    # else:
    #     self.recover_model_from_checkpoint(mode="test")
    #     self.trained_model_dir = self.config.checkpoint_dir + self.model_version_number + '/'
    #     self.evaluate_model()

    def plot_regression_predictions(self, test_ds, pred_y, manual_pred=None, i=0):

        y = [np.squeeze(y_ds) for (_, y_ds) in test_ds]
        y_flat = np.array([item for sublist in y for item in sublist])

        fig = self.plot_prediction(y_flat, pred_y, manual_pred)

        if epoch is not None:
            if i != 0:
                fig.savefig('figures/fig_{0}_{1}.png'.format(epoch, i))
            else:
                fig.savefig('figures/fig_{0}'.format(epoch))
            plt.close(fig)

        else:
            plt.show()

    def plot_and_compare_predictions(self, norm_test_ds, steps, save_figures):
        predictions = self.deep_model.predict(norm_test_ds, verbose=1, steps=steps)

        if save_figures:
            plot_regression_predictions(normalized_test_ds, predictions, epoch=self.last_epoch_number, i=fig_n)
        else:
            plot_regression_predictions(normalized_test_ds, predictions, manual_pred=manual_predictions)


    def iterate_model_output():

        return True
