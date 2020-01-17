import keras
from keras import Input, Model
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import shutil
from matplotlib import pyplot as plt

from keras.utils import plot_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 32  # 批大小
num_classes = 10  # 有多少个类别
epochs = 1  # 迭代次数
num_predictions = 20  # 预测多少个图片
input_shape = (32, 32, 3)  # 输入图片的shape
opt = RMSprop(lr=0.0001, decay=1e-6)  # 使用RMSprop优化器
output_dir = './output'  # 输出目录
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print('%s文件夹已存在，没关系，删掉了' % output_dir)
os.mkdir(output_dir)
print('%s已创建' % output_dir)

# 准备数据
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
x_train = x_train.astype('float32') / 255  # 归一化
x_val = x_val.astype('float32') / 255

# 将向量转化为二分类矩阵，也就是one-hot编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# 创建模型
input = Input(shape=input_shape)
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input)
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.25)(x)
x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.25)(x)
x = Flatten()(x)
x = Dense(units=512, activation='relu')(x)
x = Dense(units=num_classes, activation='softmax')(x)
model = Model(input, x)  # 创建模型
model.summary()  # 控制台打印模型参数信息
model_img = output_dir + '/cifar10_cnn.png'  # 模型图片的保存路径
plot_model(model, to_file=model_img, show_shapes=True)  # 模型参数信息保存为一张图片
print('%s已保存' % model_img)

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# 开始训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),  # 指定验证集
          shuffle=True)  # 混洗数据，也就是把数据打乱

# 保存整个模型
model_path = output_dir + '/keras_cifar10_trained_model.h5'
model.save(model_path)
print('%s已保存' % model_path)

# 取验证集里面的图片拿来预测看看
name = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
n = 20  # 取多少张图片
x_val = x_val[:n]
y_val = y_val[:n]

# 预测
y_predict = model.predict(x_val, batch_size=n)

# 绘制预测结果
plt.figure(figsize=(18, 3))  # 指定画布大小
for i in range(n):
    plt.subplot(2, 10, i + 1)
    plt.axis('off')  # 取消x,y轴坐标
    plt.imshow(x_val[i])  # 显示图片
    if y_val[i].argmax() == y_predict[i].argmax():
        # 如果预测正确，绿色标题
        plt.title('%s,%s' % (name[y_val[i].argmax()], name[y_predict[i].argmax()]), color='green')
    else:
        # 如果预测错误，红色标题
        plt.title('%s,%s' % (name[y_val[i].argmax()], name[y_predict[i].argmax()]), color='red')
plt.show()  # 显示画布
predict_img = output_dir + '/predict.png'
plt.savefig(predict_img)  # 保存预测图片
