"""
Load a model ckpt file, and evaluate its performance over 
internal vs external evaluation
"""
import os, numpy as np
import time
import codecs
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from simple_model import create_standard_hparams, SimpleAttentionModel
from nmt import inference, train, model_helper
from nmt.utils import vocab_utils, nmt_utils
from nmt.utils import misc_utils as utils
import iterator_utils
from evaluation_utils import shortest_neighbor_score

def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, size = 10):
    """Pick $size sentence and decode."""
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

def run_external_eval(infer_model, infer_sess, infer_iterator, infer_graph,
                model_dir, hparams, src_data, tgt_file, 
                iterator_src_placeholder,
                iterator_batch_size_placeholder,
                line_to_video):
    """
    External evaluation using the following formula:

    
    """
    with infer_graph.as_default():
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            infer_model, model_dir, infer_sess, "infer")

    dev_infer_iterator_feed_dict = {
      iterator_src_placeholder: src_data,
      iterator_batch_size_placeholder: hparams.infer_batch_size,
    }

    dev_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_iterator,
        dev_infer_iterator_feed_dict,
        tgt_file,
        "dev",
        line_to_video)

def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, label, line_to_video):
    out_dir = hparams.out_dir
    beam_width = hparams.beam_width
    metrics = hparams.metrics
    subword_option = hparams.subword_option

    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

    # First create a trans file to write output
    trans_file = os.path.join(out_dir, "output_%s" % label)

    utils.print_out("  decoding to output %s." % trans_file)

    start_time = time.time()
    num_sentences = 0
    with codecs.getwriter("utf-8")(tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
        trans_f.write("")  # Write empty string to ensure file is created.

        while True:
            try:
                nmt_outputs, _ = model.decode(sess)
                if beam_width == 0:
                    nmt_outputs = np.expand_dims(nmt_outputs, 0)

                batch_size = nmt_outputs.shape[1]
                num_sentences += batch_size

                for sent_id in range(batch_size):
                    translation = nmt_utils.get_translation(
                        nmt_outputs[0],
                        sent_id,
                        tgt_eos=hparams.eos,
                        subword_option=subword_option)
                    trans_f.write((translation + b"\n").decode("utf-8"))
            except tf.errors.OutOfRangeError:
                utils.print_time(
                    "  done, num sentences %d, num translations per input %d" %
                    (num_sentences, beam_width), start_time)
                break

    # Evaluation
    evaluation_scores = {}
    if tgt_file and tf.gfile.Exists(trans_file):
        for metric in metrics:
            score = _score_evaluate(
                tgt_file,
                trans_file,
                metric,
                line_to_video)
            evaluation_scores[metric] = score
            utils.print_out("  %s %s: %.2f" % (metric, label, score))

    return evaluation_scores

def _word_accuracy(ref_file, trans_file):
    """Compute accuracy on per word basis."""
    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "r")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "r")) as pred_fh:
            total_acc, total_count = 0., 0.
            for sentence in label_fh:
                labels = sentence.strip().split(" ")
                preds = pred_fh.readline().strip().split(" ")
                match = 0.0
                for pos in range(min(len(labels), len(preds))):
                    label = labels[pos]
                    pred = preds[pos]
                    if label == pred:
                        match += 1
                total_acc += 100 * match / max(len(labels), len(preds))
                total_count += 1
    return total_acc / total_count
    
def _neighbor(ref_file, trans_file, 
              line_to_video):
    """
    line_to_video: a function that map from a line index from ref_file to a video file (0-299)

    We should expect ref_to_video[ref_id] == trans_to_video[trans_id]

    """
    refs = []
    trans = []

    with open(ref_file, 'r') as label_fh:
        with open(trans_file, 'r') as pred_fh:
            for sentence in label_fh:
                labels = sentence.strip().split(" ")
                preds = pred_fh.readline().strip().split(" ")

                refs.append(labels)
                trans.append(preds)

    scores = []
    l1s = []
    l2s = []
    for i, (ref, tran) in enumerate(zip(refs, trans)):
        ref_video = line_to_video(i)

        score, l1, l2 = shortest_neighbor_score ( ref,  tran, ref_video )

        scores.append(score)
        l1s.append(l1)
        l2s.append(l2)

    print ('%.3f' % np.average ( l1 ))
    print ('%.3f' % np.average ( l2 ))

    return np.average ( scores )

def _score_evaluate ( ref_file, trans_file, metric, line_to_video ):
    if metric.lower() == "word_accuracy":
        evaluation_score = _word_accuracy(ref_file, trans_file)
    if metric.lower() == "neighbor":
        evaluation_score = _neighbor(ref_file, trans_file, line_to_video)

    return evaluation_score

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


        # loaded_infer_model, global_step = model_helper.create_or_load_model(
        #     infer_model, os.path.join("model", "model-4"), infer_sess, "infer")
        # # Decode 10 samples on train data
        # print ('=============================================')
        # print ('Test quality on train data')
        # _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
        #              infer_iterator, train_src_data, train_tgt_data,
        #              src_placeholder,
        #              batch_size_placeholder)
        # # Decode 10 samples on eval data
        # print ('=============================================')
        # print ('Test quality on eval data')
        # _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
        #              infer_iterator, eval_src_data, eval_tgt_data,
        #              src_placeholder,
        #              batch_size_placeholder)

        def eval_line_to_video ( line ):
            directory = line // 100

            return os.path.join('target', str(directory), str(line) + '.mp4')

        run_external_eval(infer_model, infer_sess, infer_iterator, infer_graph,
                os.path.join("model", "model-5"), hparams, eval_src_data, eval_tgt_file, 
                src_placeholder,
                batch_size_placeholder,
                eval_line_to_video)