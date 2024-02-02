import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
tfd = tfp.distributions





def dirichlet_nll_loss_with_regularization(y_true, alpha_pred, l2_lambda=1e-4):
    #for synthetic data l2_lambda=1e-4
    """
    Dirichlet Negative Log Likelihood Loss with L2 regularization.

    y_true: Tensor, true first-order distribution.
    alpha_pred: Tensor, predicted Dirichlet distributio parameters.
    l2_lambda: float, regularization strength.
    :return: Tensor, regularized negative log likelihood loss.
    """
    epsilon = 1e-6
    # Dirichlet NLL
    dist_pred = tfd.Dirichlet(1 + alpha_pred)
    # alpha_pred= alpha_pred + 1
    # dist_pred = tfd.Dirichlet(alpha_pred + 1)
    log_prob = dist_pred.log_prob(tf.clip_by_value(y_true, epsilon, 1))
    nll_loss = -tf.reduce_mean(log_prob)

    # L2 Regularization
    l2_loss = l2_lambda * tf.nn.l2_loss(alpha_pred)

    # Total Loss
    total_loss = nll_loss + l2_loss
    return total_loss


def predictor(order, feature_dim=768, dropout_rate=0.3, n_classes=3):
    """unified definition of deep model for both first and second order predictors

    Args:
        order (string): either "first" or "second"
        feature_dim (int, optional): dimesnion of input feature. Defaults to 768.
        dropout_rate (float, optional): dropout rate. Defaults to 0.3.

    Raises:
        Exception: if order is not inserted correctly. 

    Returns:
        tf.model
    """
    inputs = Input(shape=(feature_dim,))
    hidden1 = Dense(256, activation='relu')(inputs)
    hidden2 = Dense(64, activation='relu')(hidden1)
    hidden3 = Dense(16, activation='relu')(hidden2)
    drop = Dropout(rate=dropout_rate)(hidden3)
    if order == 'first':    
        outputs = Dense(n_classes, activation='softmax')(drop)
    elif order == 'second': 
        outputs = Dense(n_classes, activation='relu')(drop)
    else:
        raise Exception("order not defined")
    model = Model(inputs=inputs, outputs=outputs)

    return model 