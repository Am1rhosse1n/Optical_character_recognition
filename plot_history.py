def plot_history(netHistory):
    history = netHistory.history
    acc = history['acc']
    loss = history['loss']
    val_acc = history['val_acc']
    val_loss = history['val_loss']
    plt.xlabel('epoches')
    plt.ylabel('Error')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss','val_loss'])
    
    plt.figure()
    
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['acc','val_acc'])