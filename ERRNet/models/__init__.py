from .errnet_model import ERRNetCascadeModel, ERRNetModel, ERRNetTransformerCascadeModel

def errnet_model():
    return ERRNetModel()


def errnet_cascade_model():
    return ERRNetCascadeModel()


def errnet_transformer_cascade_model():
    return ERRNetTransformerCascadeModel()
