import tfsnippet as spt
import sys


class YourConfig(spt.Config):
    max_epoch = 1000
    learning_rate = 0.01
    activation = 'abc'


# First, you should obtain an instance of your config object
config = YourConfig()

# You can then parse config values from CLI arguments.
# For example, if sys.argv[1:] == ['--max_epoch=2000']:
from argparse import ArgumentParser
parser = ArgumentParser()
spt.register_config_arguments(config, parser)
parser.parse_args(sys.argv[1:])

# Now you can access the config value `config.max_epoch == 2000`
print(config.activation)