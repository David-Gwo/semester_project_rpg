

def experiment_builder(model_type):

    if model_type == 'speed_regression_net':
        experiments_dict = {
            "plot_predictions": {
                "ds_training_non_tensorflow": ["predict"],
                "ds_training_non_tensorflow_unnormalized": ["ground_truth"],
                "options": {
                    "output": "show",
                    "plot_data": {
                        "state_output": {
                            "type": "scalar",
                            "title": "speed prediction model",
                            "y_label": "m/s"
                        }
                    }
                }
            }
        }

    return experiments_dict

    experiments_dict = {
        "plot_predictions": {
            "ds_training_non_tensorflow": ["predict"],
            "ds_training_non_tensorflow_unnormalized": ["ground_truth"],
            "options": {
                "output": "show",
                "plot_data": {
                    # "state_output": {
                    #     "type": "10-dof-state",
                    #     "dynamic_plot": True,
                    #     "sparsing_factor": 2,
                    # },
                    "pre_integrated_p": {
                        "type": "pre_integration"
                    },
                    "pre_integrated_v": {
                        "type": "pre_integration"
                    },
                    "pre_integrated_R": {
                        "type": "pre_integration"
                    }
                }
            }
        },
        # "iterate_model_output": {
        #     "ds_training_non_tensorflow": ["predict"],
        #     "ds_training_non_tensorflow_unnormalized": ["ground_truth"],
        #     "options": {
        #         "output": "show"
        #     }
        # },
        # "training_progression": {
        #     "ds_training_non_tensorflow": ["predict"],
        #     "ds_training_non_tensorflow_unnormalized": ["ground_truth"],
        #     "options": {
        #         "output": "save",
        #         "plot_data": {
        #             "pre_integrated_p": {
        #                 "type": "pre_integration"
        #             },
        #             "pre_integrated_v": {
        #                 "type": "pre_integration"
        #             },
        #             "pre_integrated_R": {
        #                 "type": "pre_integration"
        #             }
        #         }
        #     }
        # }
    }
    return experiments_dict