# Driver file

from config import Config
from train import train_model
# from test import test_model

cfg = Config()
config = cfg.config

if config['train'] == True:
    out, loss = train_model(config)

# if config.test == 'test':
#     test_model(config)