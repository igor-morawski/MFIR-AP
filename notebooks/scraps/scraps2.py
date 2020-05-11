def build_TPA_embedding(view_id):
    # VGG-16 but dims are scaled by 1/7, only 3 blocks
    # FUTURE Think about filters -> skipping cncnts
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    # b=block c=conv m=maxpool
    # input>b1c1>b1c2>b1c3>b2m1>b2c1>b2c2>b2c3>b3m1>flatten>fc
    embedding_input = Input(shape=(None, 32, 32, 1), name='TPA{}_input'.format(view_id))
    # block1
    b1c1 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b1c1'.format(view_id))(embedding_input)
    b1c2 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b1c2'.format(view_id))(b1c1)
    b1c3 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b1c3'.format(view_id))(b1c2)
    # block2
    b2m1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)), name='TPA{}_b2m1'.format(view_id))(b1c3)
    b2c1 = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b2c1'.format(view_id))(b2m1)
    b2c2 = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b2c2'.format(view_id))(b2c1)
    b2c3 = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b2c3'.format(view_id))(b2c2)
    # block3
    b3m1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)), name='TPA{}_b3m1'.format(view_id))(b2c3)
    # FC
    flat = TimeDistributed(Flatten(), name='TPA{}_flat'.format(view_id))(b3m1)
    dense = TimeDistributed(Dense(units=EMBEDDING_UNITS, activation="relu"), name='TPA{}_dense'.format(view_id))(flat) # flatten/fc = 6.125
    embedding_output = dense
    return embedding_input, embedding_output