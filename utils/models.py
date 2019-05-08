def create_predictions_dict(predictions, model):
    n_outputs = len(model.outputs)
    predictions = {out.name.split(':')[0]: predictions[i] if n_outputs > 1 else predictions
                   for i, out in enumerate(model.outputs)}
    predictions = {i.split('/')[0]: predictions[i] for i in predictions.keys()}
    return predictions
