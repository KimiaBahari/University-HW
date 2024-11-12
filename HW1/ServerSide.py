import socket

# Server settings
host = '127.0.0.1'  # Localhost IP address
port = 12345  # Server port

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((host, port))

# Listen for incoming client connections
server_socket.listen(1)
print(f"Server is listening on {host}:{port}...")

# Accept an incoming connection from a client
client_socket, client_address = server_socket.accept()
print(f"Connection established with {client_address}.")

# Send a welcome message to the client
client_socket.send("Hello, you are connected to the server.".encode())

# Start chat communication with the client
while True:
    message = client_socket.recv(1024).decode()  # Receive message from the client
    if message.lower() == 'exit':  # If the client types 'exit', disconnect
        print("Connection closed.")
        break
    print(f"Client: {message}")
    
    response = input("You: ")  # Get the server's response
    client_socket.send(response.encode())  # Send response to the client

# Close the connection
client_socket.close()
server_socket.close()
