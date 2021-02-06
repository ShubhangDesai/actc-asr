from .BidirectionalRNN import *

def get_model(args):
    num_models = 2 if args['loss'] == 'disagreement' else 1

    models = [BidirectionalRNN(128, args['num_nodes'], args['num_layers'], 81, args['dropout'], args['layer_norm']) for _ in range(num_models)]

    if args['loss'] == 'disagreement':
        model = CoModels(models)
    else:
        model = models[0]

    return model
