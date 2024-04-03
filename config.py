import json

class ModelConfig:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            config = json.load(f)
        
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.num_layers_to_retrain = config.get('num_layers_to_retrain', 5)
        self.epochs = config.get('epochs', 10)
