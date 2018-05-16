"""
This file is a simple text to text model to translate from the instruction to commands

I will rewrite the AttentionModel to serve my format
"""
import tensorflow as tf

"""
A very simple HParams config 
that allows quick testing 
"""
def create_standard_hparams():
  return tf.contrib.training.HParams(
      # Data
      src="",
      tgt="",
      train_prefix="",
      dev_prefix="",
      test_prefix="",
      vocab_prefix="",
      embed_prefix="",
      out_dir="",

      # Networks
      num_units=512,
      num_layers=2, # not used
      num_encoder_layers=1,
      num_decoder_layers=1,
      dropout=0.2,
      unit_type="lstm",
      encoder_type="uni", # Unidirection for simplicity
      residual=False,
      time_major=True,
      num_embeddings_partitions=0,

      # Attention mechanisms
      attention="bahdanau",
      attention_architecture="standard",
      output_attention=True,
      pass_hidden_state=True,

      # Train
      optimizer="sgd",
      batch_size=32,
      init_op="uniform",
      init_weight=0.1,
      max_gradient_norm=5.0,
      learning_rate=1.0,
      warmup_steps=0,
      warmup_scheme="t2t",
      decay_scheme="luong234",
      colocate_gradients_with_ops=True,
      num_train_steps=12000,

      # Data constraints
      num_buckets=5,
      max_train=0,
      src_max_len=100,
      tgt_max_len=20,
      src_max_len_infer=100,
      tgt_max_len_infer=20,

      # Data format
      sos="<s>",
      eos="</s>",
      subword_option="",
      check_special_token=True,

      # Misc
      forget_bias=1.0,
      num_gpus=1,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=100,
      steps_per_external_eval=0,
      share_vocab=False,
      metrics=["bleu"],
      log_device_placement=False,
      random_seed=None,
      # only enable beam search during inference when beam_width > 0.
      beam_width=0,
      length_penalty_weight=0.0,
      override_loaded_hparams=True,
      num_keep_ckpts=5,
      avg_ckpts=False,

      # For inference
      inference_indices=None,
      infer_batch_size=16,
      sampling_temperature=0.0,
      num_translations_per_input=1,
  )

def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length, mode):
	"""Create attention mechanism based on the attention_option."""
	# Mechanism
	if attention_option == "luong":
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(
		    num_units, memory, memory_sequence_length=source_sequence_length)
	elif attention_option == "scaled_luong":
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(
		    num_units,
		    memory,
		    memory_sequence_length=source_sequence_length,
		    scale=True)
	elif attention_option == "bahdanau":
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
		    num_units, memory, memory_sequence_length=source_sequence_length)
	elif attention_option == "normed_bahdanau":
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
		    num_units,
		    memory,
		    memory_sequence_length=source_sequence_length,
		    normalize=True)
	else:
		raise ValueError("Unknown attention option %s" % attention_option)

	return attention_mechanism

class SimpleAttentionModel( object ):
	"""
	This class is just a monolithic class that 
	I copied from the NMT code so that we have an overview of
	how the code work.
	In this project, I will not focusing on improving anything
	related to the original NMT model, but to apply 
	a specific set of configuration on my data
	to generate a working instance.

	This model will also only use uni-direction. The reason
	is because it is not a full translation model, and biased to run
	from left to right.
	"""
	def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
		"""Create the model.

	    Args:
	      hparams: Hyperparameter configurations.
	      mode: TRAIN | EVAL | INFER
	      iterator: Dataset Iterator that feeds data.
	      source_vocab_table: Lookup table mapping source words to ids.
	      target_vocab_table: Lookup table mapping target words to ids.
	      reverse_target_vocab_table: Lookup table mapping ids to target words. Only
	        required in INFER mode. Defaults to None.
	      scope: scope of the model.
	      extra_args: model_helper.ExtraArgs, for passing customizable functions.

	    """
	    self.attention_mechanism_fn = create_attention_mechanism

	    assert isinstance(iterator, iterator_utils.BatchedInput)
	    self.iterator = iterator
	    self.mode = mode
	    self.src_vocab_table = source_vocab_table
	    self.tgt_vocab_table = target_vocab_table

	    self.single_cell_fn = None

	    # Set num layers
	    self.num_encoder_layers = hparams.num_encoder_layers
	    self.num_decoder_layers = hparams.num_decoder_layers
	    assert self.num_encoder_layers
	    assert self.num_decoder_layers

	    # Initializer
	    initializer = model_helper.get_initializer(
	        hparams.init_op, hparams.random_seed, hparams.init_weight)
	    tf.get_variable_scope().set_initializer(initializer)

	    # Embeddings
	    self.init_embeddings(hparams, scope)
	    self.batch_size = tf.size(self.iterator.source_sequence_length)

	    # Projection
	    with tf.variable_scope(scope or "build_network"):
	    	with tf.variable_scope("decoder/output_projection"):
	    		self.output_layer = layers_core.Dense(
	            	hparams.tgt_vocab_size, use_bias=False, name="output_projection")


	    # Build graph
		utils.print_out("# creating %s graph ..." % self.mode)
		dtype = tf.float32

		# res = logits, loss, final_context_state, sample_id
		with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
			# Encoder
			encoder_outputs, encoder_state = self._build_encoder(hparams)

			## Decoder
			logits, sample_id, final_context_state = self._build_decoder(
			  encoder_outputs, encoder_state, hparams)

			## Loss
			if self.mode != tf.contrib.learn.ModeKeys.INFER:
				with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
				                                           self.num_gpus)):
					loss = self._compute_loss(logits)
			else:
				loss = None

	def _build_encoder(self, hparams):
	    """Build an encoder."""
	    num_layers = self.num_encoder_layers
	    iterator = self.iterator

	    source = iterator.source
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
				    hparams, num_layers, 0)

				encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
				    cell,
				    encoder_emb_inp,
				    dtype=dtype,
				    sequence_length=iterator.source_sequence_length,
				    time_major=self.time_major,
				    swap_memory=True)
	      
	    return encoder_outputs, encoder_state

	def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
	    """Build a RNN cell with attention mechanism that can be used by decoder."""
	    attention_option = hparams.attention
	    attention_architecture = hparams.attention_architecture

	    if attention_architecture != "standard":
	    	raise ValueError(
	          "Unknown attention architecture %s" % attention_architecture)

	    num_units = hparams.num_units
	    num_layers = self.num_decoder_layers
	    beam_width = hparams.beam_width

	    dtype = tf.float32

	    # Ensure memory is batch-major
	    if self.time_major:
	    	memory = tf.transpose(encoder_outputs, [1, 0, 2])
	    else:
	    	memory = encoder_outputs

	    if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
			memory = tf.contrib.seq2seq.tile_batch(
			  memory, multiplier=beam_width)
			source_sequence_length = tf.contrib.seq2seq.tile_batch(
			  source_sequence_length, multiplier=beam_width)
			encoder_state = tf.contrib.seq2seq.tile_batch(
			  encoder_state, multiplier=beam_width)
			batch_size = self.batch_size * beam_width
	    else:
			batch_size = self.batch_size

	    attention_mechanism = self.attention_mechanism_fn(
	        attention_option, num_units, memory, source_sequence_length, self.mode)

	    cell = model_helper.create_rnn_cell(
	        unit_type=hparams.unit_type,
	        num_units=num_units,
	        num_layers=num_layers,
	        num_residual_layers=0,
	        forget_bias=hparams.forget_bias,
	        dropout=hparams.dropout,
	        num_gpus=self.num_gpus,
	        mode=self.mode,
	        single_cell_fn=self.single_cell_fn)

	    # Only generate alignment in greedy INFER mode.
	    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
	                         beam_width == 0)
	    cell = tf.contrib.seq2seq.AttentionWrapper(
	        cell,
	        attention_mechanism,
	        attention_layer_size=num_units,
	        alignment_history=alignment_history,
	        output_attention=hparams.output_attention,
	        name="attention")

	    # TODO(thangluong): do we need num_layers, num_gpus?
	    cell = tf.contrib.rnn.DeviceWrapper(cell,
	                                        model_helper.get_device_str(
	                                            num_layers - 1, self.num_gpus))

	    if hparams.pass_hidden_state:
			decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
			  cell_state=encoder_state)
	    else:
	    	decoder_initial_state = cell.zero_state(batch_size, dtype)

	    return cell, decoder_initial_state



if __init__ == '__main__':
