from numpy import ndarray


def create_predictions_dict(predicted, model):
    n_outputs = len(model.outputs)
    predicted = {
        out.name.split(':')[0]: predicted[i] if n_outputs > 1 else predicted for i, out in enumerate(model.outputs)}
    predicted = {
        i.split('/')[0]: predicted[i][0].numpy() if isinstance(predicted[i], ndarray) and predicted[i].shape == (1,)
        else predicted[i] for i in predicted.keys()}
    return predicted
