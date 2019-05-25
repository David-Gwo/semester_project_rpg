def experiment_builder(model_type):

    experiments_dict = {}
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
    elif model_type == 'windowed_integration_net':
        experiments_dict = {
            "plot_predictions": {
                "ds_testing_non_tensorflow": ["predict"],
                "ds_testing_non_tensorflow_unnormalized": ["ground_truth", "compare_prediction"],
                "options": {
                    "output": "show",
                    "plot_data": {
                        "state_output": {
                            "type": "10-dof-state",
                        }
                    }
                }
            }
        }
    elif model_type == 'windowed_integration_net_so3':
        experiments_dict = {
            # "plot_predictions": {
            #     "ds_testing_non_tensorflow": ["predict"],
            #     "ds_testing_non_tensorflow_unnormalized": ["ground_truth", "compare_prediction"],
            #     "options": {
            #         "output": "show",
            #         "plot_data": {
            #             "state_output": {
            #                 "type": "9-dof-state-lie",
            #             }
            #         }
            #     }
            # },
            "iterate_model_output": {
                "ds_training_non_tensorflow": ["predict"],
                "ds_training_non_tensorflow_unnormalized": ["ground_truth"],
                "options": {
                    "output": "show",
                    "state_out": {
                        "name": "state_output",
                        "lie": True
                    },
                    "state_in": {
                        "name": "state_input"
                    },
                    "plot_data": {
                        "state_output": {
                            "type": "9-dof-state-lie"
                        }
                    }
                }
            },
        }
    elif model_type == 'pre_integration_net':
        experiments_dict = {
            "plot_predictions": {
                "ds_testing_non_tensorflow_unnormalized": ["predict", "ground_truth", "compare_prediction"],
                "options": {
                    "output": "show",
                    "plot_data": {
                        "state_output": {
                            "type": "10-dof-state",
                            # "dynamic_plot": True,
                            "sparsing_factor": 2,
                        },
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
            }
        }
    return experiments_dict
