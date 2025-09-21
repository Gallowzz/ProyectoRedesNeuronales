import DnnLib

def init_opts(opt_name, learning_rate):
    name = opt_name.lower()
    
    if name == "sgd":
        optimizer = DnnLib.SGD(learning_rate)
        name = "SGD"
    elif opt_name == "sgd+momentum":
        optimizer = DnnLib.SGD(learning_rate, 0.9)
        name = "SGD+Momentum"
    elif opt_name == "adam":
        optimizer = DnnLib.Adam(learning_rate)
        name = "Adam"
    elif opt_name == "rmsprop":
        optimizer = DnnLib.RMSProp(learning_rate)
        name = "RMSProp"
    else:
        print("No se reconoce el Optimizador, utilizando Adam")
        name = "Adam"
        optimizer = DnnLib.Adam(learning_rate)

    return (name, optimizer)