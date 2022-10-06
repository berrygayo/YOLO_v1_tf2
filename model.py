import tensorflow as tf

# Implementation using tf.keras.applications (https://www.tensorflow.org/api_docs/python/tf/keras/applications)
# 미리 train 된 inception v3 모델을 가져온다
# & Keras Functional API (https://www.tensorflow.org/guide/keras/functional)
# keras layer 부분 쉽게 수정 가능한 사이트 , 마지막에 api 로 던져주면 사용하면 됨 
class YOLOv1(tf.keras.Model):
  def __init__(self, input_height, input_width, cell_size, boxes_per_cell, num_classes):
    super(YOLOv1, self).__init__()
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(input_height, input_width, 3))
    # include_top=False : 마지막 softmax regression layer은 빼고 가져옴, 
    # weights='imagenet' : imagenet pretrain 된 가중치 사용 
    base_model.trainable = True  # 앞부분 파라미터튜닝부분을 freeze 하지 않음 
    x = base_model.output # feature map 까지만 가져옴 

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x) # feature map을 하나의 scalar vector로 모아줌
    output = tf.keras.layers.Dense(cell_size * cell_size * (num_classes + (boxes_per_cell*5)), activation=None)(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    self.model = model
    # print model structure
    self.model.summary()

  def call(self, x):
    return self.model(x)