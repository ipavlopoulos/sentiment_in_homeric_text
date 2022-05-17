import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig
from tqdm import tqdm
import numpy as np
import pandas as pd


class GreekBERT_IB1S():
  def __init__(self,
                 trainable_layers=3,
                 max_seq_length=128,
                 show_summary=False,
                 patience=3,
                 epochs=10,
                 batch_size=32,
                 lr=2e-05,
                 session=None,
                 dense_activation = 'softmax',
                 loss='MSE',
                 monitor_loss = 'val_mse',
                 monitor_mode = 'min',
                 METRICS = [
                        tf.keras.metrics.MeanSquaredError(name="MSE"),
                        tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                        tf.keras.metrics.MeanSquaredLogarithmicError(name="MSLE"),
                  ]
                 
                 ):
        self.session = session
        self.tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1", do_lower_case=False,  max_length=max_seq_length,pad_to_max_length=True)
        self.lr = lr
        self.batch_size = batch_size
        self.trainable_layers = trainable_layers
        self.max_seq_length = max_seq_length
        self.show_summary = show_summary
        self.patience=patience
        self.epochs = epochs
        self.METRICS = METRICS
        self.loss = loss
        self.monitor_loss = monitor_loss
        self.monitor_mode = monitor_mode
        self.dense_activation = dense_activation
        self.earlystop = tf.keras.callbacks.EarlyStopping(monitor=self.monitor_loss,
                                                            patience=self.patience,
                                                            verbose=1,
                                                            restore_best_weights=True,
                                                            mode=self.monitor_mode)
        self.BERT = TFBertModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1", output_attentions = True)
        
  def to_bert_input(self, texts):
    """ This method returns the inputs that Bert needs: input_ids, input_masks, input_segments 
      :param texts: the texts that will be converted into bert inputs 
      :return: a tuple containing the input_ids, input_masks, input_segments 
    """
    input_ids, input_masks, input_segments = [],[],[]
    for text in tqdm(texts):
      inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_seq_length, pad_to_max_length=True, 
                                                  return_attention_mask=True, return_token_type_ids=True)
      input_ids.append(inputs['input_ids'])
      input_masks.append(inputs['attention_mask'])
      input_segments.append(inputs['token_type_ids'])        
    return (np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32'))
      
  def unfreeze_last_k_layers(self,k=3):
    counter = 0
    for layer in self.BERT.layers:
      for l in layer.encoder.layer:
        if counter == len(layer.encoder.layer)-k:
          return
        else:
          counter+=1
          l.trainable = False

  def build(self, bias=0):
    #encode text via bert 
    in_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_ids", dtype='int32')
    in_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_masks", dtype='int32')
    in_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="segment_ids", dtype='int32')
    bert_inputs = [in_id, in_mask, in_segment]
    bert_output = self.BERT(bert_inputs).last_hidden_state
        
    bert_output = bert_output[:,0,:] #take only the embedding of the CLS token
    x = tf.keras.layers.Dense(128, activation='tanh')(bert_output)
    pred = tf.keras.layers.Dense(4, activation=self.dense_activation, bias_initializer=tf.keras.initializers.Constant(bias))(x)
    self.model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    self.model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=self.METRICS)
    if self.show_summary:
      self.model.summary()

  def fit(self, train_texts, train_y,  dev_texts, dev_y,  bert_weights=None):
    #prepare the input for bert 
     train_input = self.to_bert_input(train_texts)
     dev_input = self.to_bert_input(dev_texts)
     self.build() #build model
     if bert_weights is not None:
      self.model.load_weights(bert_weights)
     self.model.fit(train_input,
                    train_y,
                    validation_data=(dev_input, dev_y),
                    epochs=self.epochs,
                    callbacks=[self.earlystop],
                    batch_size=self.batch_size,
                    class_weight=None 
                   )

  def predict(self, test_texts):
    #encode target
    test_input = self.to_bert_input(test_texts)
    predictions = self.model.predict(test_input)
    print('Stopped epoch: ', self.earlystop.stopped_epoch)
    return predictions

  def save_weights(self, path):
    self.model.save_weights(path)

  def load_weights(self, path):
    self.model.load_weights(path)
