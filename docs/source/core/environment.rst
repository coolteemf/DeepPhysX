CORE - Using an Environment
===========================

Dedicated code in ``Core/Environment`` and ``Core/AsyncSocket`` repositories.

Implementation
--------------

An *Environment* can be considered either as a *TcpIpClient* (communication with the *EnvironmentManager* through a
*TcpIpServer*, see section below) or not (direct communication with the *EnvironmentManager*).
Yet, the user API will be the same if the *Environment* is a *TcpIpClient* entity or not, so that the user can
implement its *Environment* regardless of this dependence.

| **Mandatory**
| Here is the list of methods that must be implemented:

:``create``: Create the simulation.
             This method is automatically called when the *Environment* component is created.

:``step``: Describe the transitions between simulation states.
           This method is automatically called when the *Pipelines* require a step in the simulation to produce a
           data sample.

| **Optional**
| Additional non mandatory methods can be implemented to describe the simulation behavior:

:``init``: Initialize the simulation.
           In some cases, the simulation components might need to be initialized in addition to being created.
           This method is automatically called when the *Environment* component is created, right after the create
           method.

:``recv_parameters``: In the case of using multiple *Environments* connected to one *TcpIpServer*, one might want to
                      parameterize each *Environment* differently to produce different kinds of samples.
                      These parameters can be set in the *EnvironmentConfig* and are then automatically sent to
                      *Environments* right after their initialization.

:``send_parameters``: On the other side, one might want to send back data from *Environments* to the
                      *EnvironmentConfig*.
                      If the method is implemented, parameters will be sent right after the previous receipt.

:``check_sample``: Each step will be followed by a sample checking.
                   This check always returns True by default, but samples can be sorted with this method and are not
                   added to the Dataset if it does not meet some criteria.

:``apply_prediction``: During the prediction *Pipeline* (and for some cases during the training *Pipeline*), the
                       *Network* predictions will be sent back to the *Environment* to be applied in the simulation.
                       This method describes how the prediction must be used in the simulation.

:``close``: For some reasons, the simulation might need to perform some tasks before being shut down.
            These tasks can be done in the close method which is automatically called before shut down.

| **Data producing**
| Some methods are dedicated to data:

:``set_training_data``: This method defines which data will be considered as training data, that is the data sent to
                        the *NetworkManager* to feed the *Network*.

:``set_loss_data``: This method allows adding specific data to compute the loss function with numeric values from the
                    simulation.

:``set_additional_dataset``: In addition to the training data, some additional data can be sent directly to the dataset
                             to replay some simulation states.
                             This method adds a new field to the *Dataset*.
                             If used, this method must be called at each step.

:``set_dataset_sample``: This method is already implemented and must not be overwritten by the user.
                         Depending on the application, when reloading existing data the *DataManager* might need to send
                         samples from the *Dataset* to the *Environment* to produce a *Network* compatible sample.
                         These samples are set in the *Environment* through this method and are named ``sample_in`` and
                         ``sample_out``.
                         In the case where the *Dataset* has additional data fields, these additional fields are named
                         ``additional_inputs`` and ``additional_outputs``.
                         Each sample can then be processed in the step function.

| **Requests**
| The *Environment* is also able to perform some requests.
| These requests are sent either directly or through a *TcpIpServer* to the EnvironmentManager*:

:``get_prediction``: Depending on the application, a prediction of the *Network* can be useful before the end of a step.
                     As the *Network* computes a prediction after a simulation step by default, the only way to have
                     this prediction during a step is to request it with this method.
                     The training data must obviously be set before triggering a prediction.

:``send_visualization``: DeepPhysX comes with an integrated *Visualizer* tool to render some components of the
                         simulation.
                         Parts of the simulation to render must be defined when creating or when initializing the
                         *Environment*.
                         An *Environment* has a visualization factory to easily create visualization data from templates
                         with ``addObject`` method: user only has to set the object type and to fill the required fields
                         to create this type of object in the *Visualizer* (see dedicated section).

:``update_visualization``: If a *Visualizer* has been created, it must be manually updated at each step by sending the
                           updated state of the simulation with this method.
                           The factory allow to easily create updated visualization data from objects ids with
                           ``updateObject_dict`` method: user only has to fill the object id (warning: the order of
                           objects is the same as the order in which they were created) and the required field (fields
                           are detailed for each object in the dedicated section).


Configuration
-------------

Using an *Environment* in one of the DeepPhysX *Pipeline* always requires an *EnvironmentConfig*.
This component’s role is both to bring together all the options for configuring an *Environment* and to either create
an instance of a single *Environment* or launch a *TcpIpServer* with several *TcpIpClients*.
In the first case, the single *Environment* will simply be created within the ``create_environment`` method, while in
the other case, the ``create_server`` method will simply create and launch a *TcpIpServer* and then start several
subprocesses, each using the ``launcherBaseEnvironment.py`` script to create and launch an *Environment* as a *Client*.

| **Simulation parameters**
| Here is a description of attributes related to a base *Configuration*:

:``environment_class``: The class from which an *Environment* will be created.
                        The attribute requires the class and not an instance, as it will be automatically created as
                        explained above.

:``visualizer``: A visualization tool is provided in DeepPhysX, which is the *VedoVisualizer*.
                 This *Visualizer* renders the specified parts of each *Environment*.
                 If no *Visualizer* is provided, the pipeline will run without any render window.

:``simulations_per_step``: The number of iterations to compute in the *Environment* at each time step.
                           An *Environment* will compute one iteration by default.

:``use_prediction_in_environment``: Each *Network* prediction will be automatically applied in the *Environment* if
                                    this flag is set to True (set to False by default).

    .. note::
        A prediction can be requested from the Environment as soon as the input data is produced, even when the step
        has not completed (see get_prediction in the section above).

| **Data parameters**
| Here is a description of attributes related to sample generation:

:``always_create_data``: This flag is useful for the training *Pipeline*.
                         If set to False (by default), the *DataManager* will request a new batch from the *Environment*
                         only for the first training epoch, then reload produced data for the remaining epochs.
                         If set to True, the *DataManager* will request a new batch from the *Environment* during the
                         entire training session.

:``screenshot_sample_rate``: This option is only available if a *Visualizer* is defined.
                             In addition to storing the produced data as *Dataset* partitions, samples can also be
                             saved as screenshots so representative ones can be easily found.
                             A screenshot of the viewer will be taken every x samples (set to 0 by default).

:``record_wrong_samples``: By default, only the good samples are stored in the *Dataset* (sorted by check_sample, see
                           the section above).
                           If this flag is set to True, the wrong samples will also be saved to dedicated partitions.

:``max_wrong_samples_per_step``: If an *Environment* produces too many wrong samples, it may be configured incorrectly.
                                 To avoid unnecessary extended data generation, a threshold can be set so that the
                                 session is stopped if too many wrong samples are produced to fill a single batch.

| **TcpIP parameters**
| Here is a description of attributes related to the *Client* configuration:

:``as_tcp_ip_client``: This flag will determine if Environments will be launched as a Client to connect to a Server (True by default) or if an Environment will be directly connected to its Manager.

    .. note::
        In the prediction *Pipeline*, only one *Environment* will be used, so the value of this flag is ignored.

:``ip_address``: Name of the IP address to bind *TcpIpObjects*.
                 The default value is “localhost” to host the *Server* and *Clients* locally.

:``port``: TCP port’s number through which *TcpIpObjects* will communicate (10000 by default).

:``environment_file``: When launching an *Environment* as a *Client*, the *EnvironmentConfig* starts a subprocess
                       involving that *Environment*.
                       To do this, the launcher will need the script in which the *Environment* is defined.
                       This script is mostly automatically detected, so this variable is non-mandatory, but in some
                       cases users may need to enter the path to their python file.

:``number_of_thread``: The number of *Environments* to launch simultaneously if the flag ``as_tcp_ip_client`` is True.

:``max_client_connection``: The maximum number of *Client*’s connections allowed by a *Server*.

:``param_dict``: *Environments* can receive additional parameters if they need to be parameterized differently.
                 These parameters are sent in the form of dictionaries by the Server when creating the *Environment*.


Client-Server Architecture
--------------------------

DeepPhysX allows users to run several *Environments* in multiprocessing to speed up data generation.
The ``AsyncSocket`` module defines a Client-Server architecture where a *TcpIpServer* communicates with several
*TcpIpClients* using a TcpIp protocol.

At the beginning of data generation, a *TcpIpServer* is launched by the *EnvironmentConfig*.
This *TcpIpServer* binds to the socket with the configured IP address on the configured port and then listens for
*TcpIpClients*.
To launch *TcpIpClients*, the *EnvironmentConfig* runs sub-processes where a launcher is called.
This launcher creates a *TcpIpClient* instance, this *TcpIpClient* is then bound to the socket with the configured IP
address on the configured port.
Once the *TcpIpClients* are all connected to the *TcpIpServer*, initialization is performed to create all the
*Environments* and initialize the parameters exchanges.
*TcpIpClients* are now ready to communicate with the *TcpIpServer* to handle the data generation.
Finally, the *TcpIpServer* triggers the shutdown of each *TcpIpClient* and closes the socket.

Both *TcpIpServer* and *TcpIpClient* inherit from *TcpIpObject* to access low levels of sending and receiving data on
the socket.
The data is sent as a custom bytes message converted with a *BytesConverter*, which handles Python common types and
NumPy arrays.
On top of these low level data exchange methods are built higher level protocols to send labeled data, labeled
dictionaries and commands.

A list of available commands is defined, *TcpIpServer* and *TcpIpClients* have then their own action implementations
to perform when receiving a command:

* A *TcpIpClient* defines the following actions to perform on commands:

  :``exit``: Set the closing flag to True to terminate the communication loop.

  :``prediction``: Receive the prediction sent by the *TcpIpServer* and apply it in the *Environment*.

  :``sample``: When using data from *Dataset*, the sample is received and defined in the *Environment* on this
               command.

  :``step``: Trigger a simulation step to produce data.
             Data should be sent to the *TcpIpServer* when the produced sample is identified as usable by sample
             checking.

* A *TcpIpServer* defines the following actions to perform on commands:

  :``prediction``: Receive data to feed the *Network*, then send back the prediction to the same *TcpIpClient*.

  :``visualization``: Receive initial or updated visualization data, then call the *Visualizer* update.
