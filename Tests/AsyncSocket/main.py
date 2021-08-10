from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from Tests.AsyncSocket.Environment import Environment


env_config = BaseEnvironmentConfig(environment_class=Environment)
env = env_config.createEnvironment()

env_config = BaseEnvironmentConfig(environment_class=Environment, number_of_thread=5)
env = env_config.createServer(batch_size=5)
