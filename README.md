# MNIST-FL-Server
## Federated Learning server for MNIST models.

The current application is the server part of Federated Learning environment implemented using Flask. The client side of the same can be found at [MNIST-FL-Client](https://github.com/nagendar-pm/MNIST-FL-Client).

The server can perform the tasks given below:
- **Train**: Trains the model it is maintaining, as per the (masked) data sent by the client
- **Update**: Updates the models maintained by the client when the clients request
