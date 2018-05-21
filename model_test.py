"""
Load a model ckpt file, and evaluate its performance over 
internal vs external evaluation
"""
import os, numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from simple_model import create_standard_hparams, SimpleAttentionModel
from nmt import inference, train, model_helper
from nmt.utils import vocab_utils, nmt_utils
from nmt.utils import misc_utils as utils
import iterator_utils

def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, size = 10):
	"""Pick a sentence and decode."""
	decode_ids = np.random.randint(0, len(src_data) - 1, size = size)
	#utils.print_out("  # %d" % decode_id)

	iterator_feed_dict = {
	  iterator_src_placeholder: [src_data[decode_id] for decode_id in decode_ids],
	  iterator_batch_size_placeholder: size,
	}
	sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

	# nmt_outputs.shape = (2, 10, 20)
	# (hparams.beam_width, size, hparams.tgt_max_len)
	nmt_outputs, _ = model.decode(sess)

	if hparams.beam_width > 0:
		# get the top translation.
		nmt_outputs = nmt_outputs[0]

	# nmt_outputs.shape = (10, 20)
	for i, decode_id in enumerate(decode_ids):
		translation = nmt_utils.get_translation(
			nmt_outputs,
			sent_id=i,
			tgt_eos=hparams.eos,
			subword_option=None)
		utils.print_out("  # %d" % decode_id)
		utils.print_out("    src: %s" % src_data[decode_id])
		utils.print_out("    ref: %s" % tgt_data[decode_id])
		utils.print_out(b"    nmt: " + translation)

if __name__ == '__main__':
    hparams = create_standard_hparams()

    DATA_DIR = 'data'

    train_src_file = os.path.join(DATA_DIR, 'instructions.txt')
    train_tgt_file = os.path.join(DATA_DIR, 'commands.txt')
    eval_src_file = os.path.join(DATA_DIR, 'eval_instructions.txt')
    eval_tgt_file = os.path.join(DATA_DIR, 'eval_commands.txt')
    src_vocab_file = os.path.join(DATA_DIR,'instructions.vocab')
    tgt_vocab_file = os.path.join(DATA_DIR,'commands.vocab')

    # list of sentences
    train_src_data = inference.load_data(train_src_file)
    train_tgt_data = inference.load_data(train_tgt_file)
    eval_src_data = inference.load_data(eval_src_file)
    eval_tgt_data = inference.load_data(eval_tgt_file)

    ### Set vocab files
    src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
        src_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)
    tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
        tgt_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
    hparams.add_hparam("src_vocab_file", src_vocab_file)
    hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)
    # There are no pretrained embeddings:
    hparams.add_hparam("src_embed_file", "")
    hparams.add_hparam("tgt_embed_file", "")


    out_dir = hparams.out_dir

    infer_graph = tf.Graph()
    infer_sess = tf.Session(graph=infer_graph)

    # Summary writer
    summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, 'summary'), infer_graph)

    with infer_graph.as_default(), tf.container("infer"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)

        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file, default_value=vocab_utils.UNK)

        # src_placeholder is placeholder for list of sentences
        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        # The reason we do not use tf.data.TextLineDataset
        # is because we can feed different files
        src_dataset = tf.data.Dataset.from_tensor_slices(
            src_placeholder)

        # Infer iterator only iterates over source dataset
        infer_iterator = iterator_utils.get_infer_iterator(
            src_dataset,
            src_vocab_table,
            batch_size=batch_size_placeholder,
            eos=hparams.eos,
            src_max_len=hparams.src_max_len_infer)


        infer_model = SimpleAttentionModel(
            hparams,
            reverse_target_vocab_table = reverse_tgt_vocab_table,
            iterator=infer_iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table)


        loaded_infer_model, global_step = model_helper.create_or_load_model(
            infer_model, os.path.join("model", "model-2"), infer_sess, "infer")

        # Decode 10 samples on train data
        print ('=============================================')
        print ('Test quality on train data')
        _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                     infer_iterator, train_src_data, train_tgt_data,
                     src_placeholder,
                     batch_size_placeholder)

        # Decode 10 samples on eval data
        print ('=============================================')
        print ('Test quality on eval data')
        _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                     infer_iterator, eval_src_data, eval_tgt_data,
                     src_placeholder,
                     batch_size_placeholder)