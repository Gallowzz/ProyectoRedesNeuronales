import DnnLib

def init_opts(opt_name, learning_rate):
    name = opt_name.lower()
    
    if name == "sgd":
        optimizer = DnnLib.SGD(learning_rate)
        name = "SGD"
    elif name == "sgd+momentum":
        optimizer = DnnLib.SGD(learning_rate, 0.9)
        name = "SGD+Momentum"
    elif name == "adam":
        optimizer = DnnLib.Adam(learning_rate)
        name = "Adam"
    elif name == "rmsprop":
        optimizer = DnnLib.RMSprop(learning_rate)
        name = "RMSprop"
    else:
        print("No se reconoce el Optimizador, utilizando Adam")
        name = "Adam"
        optimizer = DnnLib.Adam(learning_rate)

    return (name, optimizer)