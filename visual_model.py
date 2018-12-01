import os
import time
import tensorflow as tf
from nmt.utils import vocab_utils
import iterator_utils
from nmt import model_helper
from nmt.utils import misc_utils as utils
from tensorflow.python.layers.core import Dense
import helper as helper_utils
from simple_model import SimpleAttentionModel

class VisualModel ( SimpleAttentionModel ):
    def _build_encoder(self, hparams):
        """
        Replace _build_encoder of SimpleAttentionModel.

        The most important difference is that we have two 
        sequential inputs instead of one
        """
        num_layers = self.num_encoder_layers
        iterator = self.iterator

        textual_source = iterator.source
        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [max_time, batch_size, num_units]
            encoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_encoder, source)

            # Encoder_outputs: [max_time, batch_size, num_units]
            # We only have uni type
            if hparams.encoder_type == "uni":
                utils.print_out("  num_layers = %d" % num_layers)
                cell = self._build_encoder_cell(
                    hparams, num_layers)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inp,
                    dtype=dtype,
                    sequence_length=iterator.source_sequence_length,
                    time_major=self.time_major,
                    swap_memory=True)
          
        return encoder_outputs, encoder_state

	def _build_decoder(self, encoder_outputs, encoder_state, hparams):
        """Build and run a RNN decoder with a final projection layer.

        Args:
          encoder_outputs: The outputs of encoder for every time step.
          encoder_state: The final state of the encoder.
          hparams: The Hyperparameters configurations.

        Returns:
          A tuple of final logits and final decoder state:
            logits: size [time, batch_size, vocab_size] when time_major=True.
        """

        ### Start and end of sequence 
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                             tf.int32)
        iterator = self.iterator

        maximum_iterations = hparams.tgt_max_len_infer
        utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)

        ## Decoder.
        with tf.variable_scope("decoder") as decoder_scope:
            # decoder_initial_state is basically zeros
            # This is different from encoder-decoder framework
            # in which state of encoder is passed into decoder

            # This part is just a reflection of how lazy I am
            # Instead of having different values of num_units
            # One for encoder, one for decoder
            # I just use one, and change it accordingly
            # before passing into _build_decoder_cell
            hparams.num_units += hparams.visual_size

            cell, decoder_initial_state = self._build_decoder_cell(
              hparams, encoder_outputs, encoder_state,
              iterator.source_sequence_length)

            # Now preserve it, in case we might have to use it again
            hparams.num_units -= hparams.visual_size

            ## Train or eval
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                # decoder_emp_inp: [max_time, batch_size, num_units]
                target_input = iterator.target_input
                # target_visual: [max_time, batch_size, visual_size]
                target_visual = iterator.target_visual

                if self.time_major:
                    target_input = tf.transpose(target_input)
                    target_visual = tf.transpose(target_visual)

                # decoder_emb_inp.get_shape() = [max_time, batch_size, num_units]
                decoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_decoder, target_input)

                # concatenated_input.get_shape() = [max_time, batch_size, 
                #                                   num_units + visual_size]
                concatenated_input = tf.concat ([decoder_emb_inp, target_visual], 
                						axis = 2)

                # Helper
                helper = helper_utils.TrainingHelper(
                    concatenated_input, iterator.target_sequence_length,
                    time_major=self.time_major)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state,)

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

                sample_id = outputs.sample_id

                # Note from original code: there's a subtle difference here between train and inference.
                # We could have set output_layer when create my_decoder
                #   and shared more code between train and inference.
                # We chose to apply the output_layer to all timesteps for speed:
                #   10% improvements for small models & 20% for larger ones.
                # If memory is a concern, we should apply output_layer per timestep.
                # Tuan's note: self.output_layer is a Dense layer predicting 
                # an output with a number of predicting classes.
                # outputs.rnn_output has a size of [time, batch_size, cell_size]
                logits = self.output_layer(outputs.rnn_output)

            ## Inference
            else:
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight)

                else:
                    # Beam_width might not be very important in this problem
                    # But I should include it to make a comparison to the reinforcement
                    # learning model
                    # Helper
                    sampling_temperature = hparams.sampling_temperature

                    # Uses sampling (from a distribution) instead of argmax and 
                    # passes the result through an embedding layer to get the next input.
                    # sampling_temperature control the level of randomness (or argmax*ness*)
                    if sampling_temperature > 0.0:
                        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                            self.embedding_decoder, start_tokens, end_token,
                            softmax_temperature=sampling_temperature,
                            seed=hparams.random_seed)
                    else:
                        helper = helper_utils.ControllerGreedyEmbeddingHelper(
                            self.embedding_decoder, start_tokens, end_token)

                    # Decoder
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell,
                        helper,
                        decoder_initial_state,
                        output_layer=self.output_layer  # applied per timestep
                    )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=maximum_iterations,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    # This logits has been run through the dense self.output_layer
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state