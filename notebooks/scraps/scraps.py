
    '''
    FUNCTIONAL
    # VGG-16 but dims are scaled by 1/7, only 3 blocks
    # FUTURE Think about filters -> skipping cncnts
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    # b=block c=conv m=maxpool
    # input>b1c1>b1c2>b1c3>b2m1>b2c1>b2c2>b2c3>b3m3>flatten>fc

    # block1
    b1c1 = Conv2D(input_shape=(32, 32, 1), filters=64, kernel_size=(
        3, 3), padding="same", activation="relu")(input)
    b1c2 = Conv2D(filters=64, kernel_size=(3, 3),
                  padding="same", activation="relu")(b1c1)
    b1c3 = Conv2D(filters=64, kernel_size=(3, 3),
                  padding="same", activation="relu")(b1c2)
    # block2
    b2m1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b1c3)
    b2c1 = Conv2D(filters=128, kernel_size=(3, 3),
                  padding="same", activation="relu")(b2m1)
    b2c2 = Conv2D(filters=128, kernel_size=(3, 3),
                  padding="same", activation="relu")(b2c1)
    b2c3 = Conv2D(filters=128, kernel_size=(3, 3),
                  padding="same", activation="relu")(b2c2)
    # block3
    b3m3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(b2c3)
    # FC
    flatten = Flatten()(b3m3)
    fc = Dense(units=1024, activation="relu")(flatten)  # flatten/fc = 6.125
    # output
    # input>b1c1>b1c2>b1c3>b2m1>b2c1>b2c2>b2c3>b3m3>flatten>fc
    model = Model(inputs=input, outputs=fc)
    return model
    '''