from server import TcpIpServerAsync
import os
import sys
import threading
import subprocess


if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <nb_clients> <batch_size>")
    sys.exit(1)


def start_server(max_clients, nb_batches, batch_sizes):
    TcpIpServerAsync(max_client_count=max_clients, nb_batches=nb_batches, batch_size=batch_sizes,
                     nb_client=nb_client_connections)


def start_client(idx):
    subprocess.run(["python3", os.path.join(os.path.dirname(os.path.realpath(__file__)), 'client.py'), str(idx)])


# Tests parameters
max_client_connections = 10
nb_client_connections = int(sys.argv[1])
batch_size = int(sys.argv[2])
nb_batch = 5

# Start server in a dedicated thread
server_thread = threading.Thread(target=start_server, args=(max_client_connections, nb_batch, batch_size))
res = server_thread.start()

# Start clients in dedicated subprocesses
client_threads = []
for i in range(nb_client_connections):
    client_thread = threading.Thread(target=start_client, args=(i,))
    client_threads.append(client_thread)
for client in client_threads:
    client.start()

print("End of the main")
