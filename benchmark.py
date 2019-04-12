import argparse
import os
import re
import time
from time import sleep
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm
from tacotron.utils.text import text_to_sequence
import numpy as np
from datasets import audio
import librosa 

checkpoint_path = "logs-Tacotron/taco_pretrained/tacotron_model.ckpt-200000"
synth = Synthesizer()

synth.load(checkpoint_path, hparams)

# save_dir='.'
# tf.train.write_graph(synth.session.graph.as_graph_def(), '.', os.path.join(save_dir,'tacotron_model.pbtxt'), as_text=True)
# tf.train.write_graph(synth.session.graph.as_graph_def(), '.', os.path.join(save_dir,'tacotron_model.pb'), as_text=False)




texts = ['Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!']
texts = [ "It took me quite a long time to develop a voice. Now that I have it I am not going to be silent." ]
texts = ["Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase the grey matter in the parts of the brain responsible for emotional regulation and learning . "]

texts1 = ['''It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.
“My dear Mr. Bennet,” said his lady to him one day, “have you heard that Netherfield Park is let at last?”
Mr. Bennet replied that he had not.
“But it is,” returned she; “for Mrs. Long has just been here, and she told me all about it.”
Mr. Bennet made no answer.
“Do you not want to know who has taken it?” cried his wife impatiently.
“You want to tell me, and I have no objection to hearing it.”
This was invitation enough.
“Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week.”
“What is his name?”
“Bingley.”
“Is he married or single?”
“Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!”
“How so? How can it affect them?”
“My dear Mr. Bennet,” replied his wife, “how can you be so tiresome! You must know that I am thinking of his marrying one of them.”
“Is that his design in settling here?”
“Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes.”
“I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley may like you the best of the party.”
“My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary now. When a woman has five grown-up daughters, she ought to give over thinking of her own beauty.”
“In such cases, a woman has not often much beauty to think of.”
“But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood.”
“It is more than I engage for, I assure you.”
“But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account, for in general, you know, they visit no newcomers. Indeed you must go, for it will be impossible for us to visit him if you do not.”
“You are over-scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying whichever he chooses of the girls; though I must throw in a good word for my little Lizzy.”
“I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference.”
“They have none of them much to recommend them,” replied he; “they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters.”
“Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no compassion for my poor nerves.”
“You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these last twenty years at least.”
“Ah, you do not know what I suffer.”
“But I hope you will get over it, and live to see many young men of four thousand a year come into the neighbourhood.”
“It will be no use to us, if twenty such should come, since you will not visit them.”
“Depend upon it, my dear, that when there are twenty, I will visit them all.”
Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of three-and-twenty years had been insufficient to make his wife understand his character. Her mind was less difficult to develop. She was a woman of mean understanding, little information, and uncertain temper. When she was discontented, she fancied herself nervous. The business of her life was to get her daughters married; its solace was visiting and news.
''']
cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
input_lengths = [len(seq) for seq in seqs]
max_seq_len=20000

feed_dict = {
    synth.inputs: seqs,
    synth.input_lengths: np.asarray(input_lengths, dtype=np.int32),
    synth.split_infos : np.asarray([[max_seq_len, 0, 0, 0]], dtype=np.int32),
}

t_1 = time.time()
mels, alignments, stop_tokens = synth.session.run([synth.mel_outputs, synth.alignments, synth.stop_token_prediction], feed_dict=feed_dict)
print(" >  Run-time: {}".format(time.time() - t_1))

import sys
sys.path.insert(0,'../WaveRNN-Pytorch/lib/build-src-RelDebInfo')
sys.path.insert(0,'../WaveRNN-Pytorch/library/build-src-Desktop-RelWithDebInfo')
import WaveRNNVocoder

vocoder=WaveRNNVocoder.Vocoder()

vocoder.loadWeights('../WaveRNN-Pytorch/model_outputs/model.bin')

mm=mels[0].squeeze()

mm[mm<-4]=-4
mm[mm>4]=4
wav = vocoder.melToWav(mels[0].squeeze().T)

#wav=audio.inv_mel_spectrogram(mels[0].squeeze().T, hparams)
librosa.output.write_wav('test.wav', wav, 16000)
plt.plot(wav)
print()

#%%
from google.protobuf import text_format
from tensorflow.python import pywrap_tensorflow
checkpoint_path = "/home/eugening/Neural/MachineLearning/Speech/Tacotron-2/checkpoints/tacotron_model.ckpt-80000"

var_list = {}
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()


#%%
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph

def freeze_model(saved_model_dir, output_node_names, output_filename):
    output_graph_filename = os.path.join(saved_model_dir, output_filename)
    initializer_nodes = ''
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags = tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )
    print('graph freezed!')

freeze_model("checkpoints/tacotron_model.ckpt", "mel_outputs", "tactron2_frozen.pb")

freeze_graph.freeze_graph("checkpoints/tacotron_model.pb", "", True, "checkpoints/tacotron_model.ckpt-80000",
                          "mel_outputs", "", "", "checkpoints/tactron2_frozen.pb", True, "")