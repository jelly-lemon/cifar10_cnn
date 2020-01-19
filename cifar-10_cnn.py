import os
import shutil

from keras import Input, Model
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model
from matplotlib import pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用哪个GPU进行训练

epochs = 5  # 迭代次数
batch_size = 32  # 批大小
opt = RMSprop(lr=0.0001, decay=1e-6)  # 使用RMSprop优化器
num_classes = 10  # 有多少个类别
input_shape = (32, 32, 3)  # 图片的shape
output_dir = './output'  # 输出目录
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print('%s文件夹已存在，但是没关系，我们删掉了' % output_dir)
os.mkdir(output_dir)
print('%s已创建' % output_dir)

# 准备数据
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
x_train = x_train.astype('float32') / 255  # 归一化
x_val = x_val.astype('float32') / 255

# 将向量转化为二分类矩阵，也就是one-hot编码
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

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
model.summary()  # 打印到控制台查看模型参数信息
model_img = output_dir + '/cifar10_cnn.png'  # 模型结构图保存路径
plot_model(model, to_file=model_img, show_shapes=True) # 模型结构保存为一张图片
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

# 保存模型
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
        # 预测正确，用绿色标题
        plt.title('%s,%s' % (name[y_val[i].argmax()], name[y_predict[i].argmax()]), color='green')
    else:
        # 预测错误，用红色标题
        plt.title('%s,%s' % (name[y_val[i].argmax()], name[y_predict[i].argmax()]), color='red')
plt.show()  # 显示画布
predict_img = output_dir + '/predict.png'
print('%s已保存' % predict_img)
plt.savefig(predict_img)  # 保存预测图片
