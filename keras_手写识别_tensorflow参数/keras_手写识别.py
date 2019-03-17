import keras
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten,BatchNormalization
from keras import backend as K
import numpy as np
import struct
import time

def model_write(input_shape):
    x_input = Input(shape=input_shape)
    X = Conv2D(32,kernel_size=(5,5),strides=(1,1))(x_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)

    X = Conv2D(64,kernel_size=(5,5),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)

    X = Dropout(0.5)(X)
    X = Flatten()(X)
    Y = Dense(10,activation='softmax')(X)

    model = keras.Model(inputs=x_input,outputs=Y,name='model_write')
    return model

# 训练集文件
train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # print('data:',bin_data)
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            pass
            # print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            pass
            # print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    train_images = np.expand_dims(train_images,axis=-1)
    test_images = np.expand_dims(test_images,axis=-1)
    print(train_images.shape)
    print(train_labels.shape)
    train_ho_labels = keras.utils.to_categorical(train_labels, num_classes=10)
    test_ho_labels = keras.utils.to_categorical(test_labels, num_classes=10)
    # print(train_ho_labels)
    # print(test_ho_labels)
    return train_images,train_ho_labels,test_images,test_ho_labels

if __name__ == '__main__':
    print(time.time())
    train_images, train_ho_labels, test_images, test_ho_labels = run()
    model = model_write((28, 28,1))
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_ho_labels, epochs=200, batch_size=256)
    pred = model.evaluate(test_images,test_ho_labels)
    # predict = model.predict(test_images)

    print(pred[0],pred[1])
    # print(predict)
    print(time.clock())




