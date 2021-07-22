import os
import numpy as np
import multiprocessing as mp

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class EnvironmentManager:

    def __init__(self, environment_config: BaseEnvironmentConfig, session_dir=None):
        """

        :param BaseEnvironmentConfig environment_config:
        :param str session_dir: Name of the directory in which to write all of the neccesary data
        """
        self.session_dir = session_dir
        self.number_of_thread = environment_config.number_of_thread
        self.multiprocessMethod = environment_config.multiprocess_method
        # Create single or multiple environments according to multiprocessing value
        self.environment = environment_config.createEnvironment()

        self.always_create_data = environment_config.always_create_data

    def getData(self, batch_size, get_inputs, get_outputs, animate):
        # Getting data from single environment
        if self.number_of_thread == 1:
            inputs, outputs = self.computeSingleThreadInputOutputFromEnvironment(batch_size, get_inputs, get_outputs,
                                                                                 animate)
        # Getting data from multiple environments
        else:
            inputs = np.empty((batch_size, *self.environment.input_size))
            outputs = np.empty((batch_size, *self.environment.output_size))
            """if self.multiprocessMethod == 'process':
                inputs, outputs = self.computeMultipleProcess(batch_size, get_inputs, get_outputs)
            else:
                inputs, outputs = self.computeMultiplePool(batch_size, get_inputs, get_outputs)"""
        return {'in': inputs, 'out': outputs}

    def computeSingleThreadInputOutputFromEnvironment(self, batch_size, get_inputs, get_outputs, animate):
        """
        Compute a batch of data from the environment

        :param int batch_size: Size of a batch
        :param bool get_inputs: If True compute and return input
        :param bool get_outputs: If True compute an return output
        :param bool animate: If True run environment step

        :return: dict of format {'in': numpy.ndarray, 'out': numpy.ndarray}
        """

        if get_inputs:
            inputs = np.empty((0, *self.environment.input_size))

            def input_condition(input_tensor):
                return input_tensor.shape[0] <= batch_size
        else:
            inputs = np.array([])

            def input_condition(input_tensor):
                return True

        if get_outputs:
            outputs = np.empty((0, *self.environment.output_size))

            def output_condition(output_tensor):
                return output_tensor.shape[0] <= batch_size
        else:
            outputs = np.array([])

            def output_condition(output_tensor):
                return True

        while input_condition(inputs) and output_condition(outputs):
            if animate:
                for _ in range(self.environment.simulations_per_step):
                    self.environment.step()

            if get_inputs:
                self.environment.computeInput()
                if self.environment.checkSample(check_input=get_inputs, check_output=False):
                    inputs = np.concatenate((inputs, self.environment.getInput()))
                else:
                    self.environment.save_wrong_sample(self.session_dir)

            if get_outputs:
                self.environment.computeOutput()
                if self.environment.checkSample(check_input=False, check_output=get_outputs):
                    outputs = np.concatenate((outputs, self.environment.getOutput()))
                else:
                    self.environment.save_wrong_sample(self.session_dir)
        return {'in': inputs, 'out': outputs}

    """def computeMultipleProcess(self, batch_size, get_inputs, get_outputs):
        inputs = np.empty((batch_size, self.environment[0].inputSize))
        outputs = np.empty((batch_size, self.environment[0].outputSize))
        produced_samples = 0
        while produced_samples < batch_size:
            process_list = []
            parent_conn_list = []
            nb_samples = min(self.number_of_thread, batch_size - produced_samples)
            # Start processes
            for i in range(nb_samples):
                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(target=self.processStep, args=(self.environment[i], child_conn,))
                p.start()
                process_list.append(p)
                parent_conn_list.append(parent_conn)
            # Synchronize processes
            for i in range(nb_samples):
                process_list[i].join()
            # Get data
            for i in range(nb_samples):
                self.environment[i] = parent_conn_list[i].recv()
                if get_inputs:
                    inputs[produced_samples + i] = self.environment[i].getInput()
                if get_outputs:
                    outputs[produced_samples + i] = self.environment[i].getOutput()
            produced_samples += nb_samples
        return inputs, outputs

    def processStep(self, env, conn):
        for _ in range(env.simulationsPerStep):
            env.step()
        conn.send(env)
        conn.close()

    def computeMultiplePool(self, batch_size, get_inputs, get_outputs):
        inputs = np.empty((batch_size, self.environment[0].inputSize))
        outputs = np.empty((batch_size, self.environment[0].outputSize))
        produced_samples = 0
        while produced_samples < batch_size:
            nb_samples = min(self.number_of_thread, batch_size - produced_samples)
            # Start pool
            with mp.Pool(processes=nb_samples) as pool:
                self.environment[:nb_samples] = pool.map(self.poolStep, self.environment[:nb_samples])
                pool.close()
                pool.join()
            # Get data
            for i in range(nb_samples):
                if get_inputs:
                    inputs[produced_samples + i] = self.environment[i].getInput()
                if get_outputs:
                    outputs[produced_samples + i] = self.environment[i].getOutput()
            produced_samples += nb_samples
        return inputs, outputs

    def poolStep(self, env):
        for _ in range(env.simulationsPerStep):
            env.step()
        return env"""

    def close(self):
        """
        Close the environment

        :return:
        """
        self.environment.close()

    def step(self, environment=None):
        """
        Run a step of environment

        :param BaseEnvirnment environment: Environment with an implement step function

        :return:
        """
        if environment is None:
            for _ in range(self.environment.simulations_per_step):
                self.environment.step()
        else:
            for _ in range(environment.simulations_per_step):
                environment.step()
