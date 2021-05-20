import os
import numpy as np
import multiprocessing as mp

from DeepPhysX.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig


class EnvironmentManager:

    def __init__(self, environment_config: BaseEnvironmentConfig, trainer):
        self.environmentConfig = environment_config
        self.multiprocessing = environment_config.multiprocessing
        self.multiprocessMethod = environment_config.multiprocessMethod
        # Create single or multiple environments according to multiprocessing value
        self.environment = environment_config.createEnvironment(trainer)

    def getData(self, batch_size, get_inputs, get_outputs, animate):
        # Getting data from single environment
        if self.multiprocessing == 1:
            inputs, outputs = self.computeSingleEnvironment(batch_size, get_inputs, get_outputs, animate)
        # Getting data from multiple environments
        else:
            inputs = np.empty((batch_size, *self.environment.inputSize))
            outputs = np.empty((batch_size, *self.environment.outputSize))
            """if self.multiprocessMethod == 'process':
                inputs, outputs = self.computeMultipleProcess(batch_size, get_inputs, get_outputs)
            else:
                inputs, outputs = self.computeMultiplePool(batch_size, get_inputs, get_outputs)"""
        return {'in': inputs, 'out': outputs}

    def computeSingleEnvironment(self, batch_size, get_inputs, get_outputs, animate):
        inputs = np.empty((batch_size, *self.environment.inputSize))
        outputs = np.empty((batch_size, *self.environment.outputSize))
        i = 0
        while i < batch_size:
            if animate:
                for _ in range(self.environment.simulationsPerStep):
                    self.environment.step()
            if get_inputs:
                self.environment.computeInput()
            if get_outputs:
                self.environment.computeOutput()
            if self.environment.checkSample(check_input=get_inputs, check_output=get_outputs):
                inputs[i] = self.environment.getInput()
                outputs[i] = self.environment.getOutput()
                i += 1
        if get_inputs:
            inputs = self.environment.transformInputs(inputs)
        if get_outputs:
            outputs = self.environment.transformOutputs(outputs)
        return inputs, outputs

    """def computeMultipleProcess(self, batch_size, get_inputs, get_outputs):
        inputs = np.empty((batch_size, self.environment[0].inputSize))
        outputs = np.empty((batch_size, self.environment[0].outputSize))
        produced_samples = 0
        while produced_samples < batch_size:
            process_list = []
            parent_conn_list = []
            nb_samples = min(self.multiprocessing, batch_size - produced_samples)
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
            nb_samples = min(self.multiprocessing, batch_size - produced_samples)
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
        # Todo : delete environments
        pass



    def step(self, environment=None):
        # Todo : not multiprocessing friendly...
        if environment is None:
            for _ in range(self.environment.simulationsPerStep):
                self.environment.step()
        else:
            for _ in range(environment.simulationsPerStep):
                environment.step()

