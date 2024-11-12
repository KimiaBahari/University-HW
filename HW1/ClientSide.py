import socket

# Client settings
host = '127.0.0.1'  # Server IP address
port = 12345  # Server port

# Create a TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((host, port))

# Receive and print the welcome message from the server
welcome_message = client_socket.recv(1024).decode()
print(welcome_message)

# Start chat communication with the server
while True:
    message = input("You: ")  # Get the user's message
    client_socket.send(message.encode())  # Send the message to the server
    
    if message.lower() == 'exit':  # If the user types 'exit', disconnect
        print("Connection closed.")
        break
    
    response = client_socket.recv(1024).decode()  # Receive the server's response
    print(f"Server: {response}")

# Close the connection
client_socket.close()
