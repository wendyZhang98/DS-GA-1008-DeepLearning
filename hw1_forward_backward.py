import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()
  
    def d_nonlinearity(self, kind, x, out):
        if kind == "relu":
            return (x > 0).type(x.dtype)
        elif kind == "sigmoid":
            return out * (1 - out)
        else:
            return torch.ones_like(x)

    def nonlinearity(self, kind, x):
        if kind == "relu":
            return torch.maximum(torch.zeros_like(x), x)
        elif kind == "sigmoid":
            return torch.sigmoid(x)
        else:
            return x

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # (batch_size, linear_1_out_features)
        z1 = x @ self.parameters["W1"].T + self.parameters["b1"]
        # (batch_size, linear_1_out_features)
        z2 = self.nonlinearity(self.f_function, z1)
        # (batch_size, linear_2_out_features)
        z3 = z2 @ self.parameters["W2"].T + self.parameters["b2"]
        # (batch_size, linear_2_out_features)
        yhat = self.nonlinearity(self.g_function, z3)

        self.cache["X"] = x
        self.cache["z1"] = z1
        self.cache["z2"] = z2
        self.cache["z3"] = z3
        self.cache["yhat"] = yhat

        return yhat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # (batch_size, linear_2_out_features)
        dyhat_dz3 = self.d_nonlinearity(self.g_function,
                                        self.cache["z3"],
                                        self.cache["yhat"])
        # (batch_size, linear_2_out_features)
        dJ_dz3 = dJdy_hat * dyhat_dz3
        # (linear_2_out_features, linear_1_out_features)
        dJ_dW2 = dJ_dz3.T @ self.cache["z2"]
        # (linear_2_out_features,)
        dJ_db2 = dJ_dz3.T @ torch.ones(dJ_dz3.shape[0])
        # (batch_size, linear_1_out_features)
        dJ_dz2 = dJ_dz3 @ self.parameters["W2"]
        # (batch_size, linear_1_out_features)
        dz2_dz1 = self.d_nonlinearity(self.f_function,
                                      self.cache["z1"],
                                      self.cache["z2"])
        # (batch_size, linear_1_out_features)
        dJ_dz1 = dJ_dz2 * dz2_dz1
        # (linear_1_out_features, linear_1_in_features)
        dJ_dW1 = dJ_dz1.T @ self.cache["X"]
        # (linear_1_out_features,)
        dJ_db1 = dJ_dz1.T @ torch.ones(dJ_dz1.shape[0])

        self.grads["dJdW1"] = dJ_dW1
        self.grads["dJdb1"] = dJ_db1
        self.grads["dJdW2"] = dJ_dW2
        self.grads["dJdb2"] = dJ_db2

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    norm_const = y.shape[0]*y.shape[1]
    J = torch.mean(torch.square(y_hat-y))
    dJdy_hat = 2*(y_hat-y)/norm_const

    return J, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    norm_const = y.shape[0]*y.shape[1]
    loss = -torch.mean(y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat))
    dJdy_hat = (-(y/y_hat) + ((1-y)/(1-y_hat)))/norm_const

    return loss, dJdy_hat
