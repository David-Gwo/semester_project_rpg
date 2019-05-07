def create_predictions_dict(predictions, model):
    predictions = {out.name.split(':')[0]: predictions[i] for i, out in enumerate(model.outputs)}
    predictions = {i.split('/')[0]: predictions[i] for i in predictions.keys()}
    return predictions
