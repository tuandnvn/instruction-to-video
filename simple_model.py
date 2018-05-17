"""
This file is a simple text to text model to translate from the instruction to commands

I will rewrite the AttentionModel to serve my format
"""
import tensorflow as tf
from nmt.utils import vocab_utils
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
      learning_rate=1.0,
      warmup_steps=0,
      warmup_scheme="t2t",
      decay_scheme="luong234",
      colocate_gradients_with_ops=True,
      num_train_steps=1000,

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
      beam_width=2,
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
		# final_context_state are the context 
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

		if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
			self.train_loss = loss
		elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
			self.eval_loss = loss
		elif self.mode == tf.contrib.learn.ModeKeys.INFER:
			self.final_context_state =  final_context_state
			self.sample_words = reverse_target_vocab_table.lookup(
			  tf.to_int64(sample_id))

		self.global_step = tf.Variable(0, trainable=False)
    	params = tf.trainable_variables()

    	# Gradients and SGD update operation for training the model.
		# Arrage for the embedding vars to appear at the beginning.
		if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
			self.learning_rate = tf.constant(hparams.learning_rate)
			# warm-up
			self.learning_rate = self._get_learning_rate_warmup(hparams)
			# decay
			self.learning_rate = self._get_learning_rate_decay(hparams)

			# Optimizer
			if hparams.optimizer == "sgd":
				opt = tf.train.GradientDescentOptimizer(self.learning_rate)
				tf.summary.scalar("lr", self.learning_rate)
			elif hparams.optimizer == "adam":
				opt = tf.train.AdamOptimizer(self.learning_rate)

			self.update = opt.minimize(self.train_loss)

			# Summary
			self.train_summary = tf.summary.merge([
			  	tf.summary.scalar("lr", self.learning_rate),
			  	tf.summary.scalar("train_loss", self.train_loss),
			])

		if self.mode == tf.contrib.learn.ModeKeys.INFER:
      		self.infer_summary = tf.no_op()

      	# Saver
	    self.saver = tf.train.Saver(
	        tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

	    # Print trainable variables
	    utils.print_out("# Trainable variables")
	    for param in params:
			utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

	
	##############################  Public methods  ###############################

	def train(self, sess):
	    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
	    return sess.run([self.update,
	                     self.train_loss,
	                     self.train_summary,
	                     self.global_step,
	                     self.batch_size,
	                     self.grad_norm,
	                     self.learning_rate])

	def eval(self, sess):
	    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
	    return sess.run([self.eval_loss,
	                     self.batch_size])

	def infer(self, sess):
	    assert self.mode == tf.contrib.learn.ModeKeys.INFER
	    return sess.run([
	        self.infer_summary, self.sample_words
	    ])

	def decode(self, sess):
		"""Decode a batch.

		Args:
		  sess: tensorflow session to use.

		Returns:
		  A tuple consiting of outputs, infer_summary.
		    outputs: of size [batch_size, time] if beam_width == 0
		    		of of size [batch_size, time, beam_width]
		"""
		infer_summary, sample_words = self.infer(sess)

		# make sure outputs is of shape [batch_size, time] or [beam_width,
		# batch_size, time] when using beam search.
		if self.time_major:
			sample_words = sample_words.transpose()
		elif sample_words.ndim == 3:  # beam search output in [batch_size,
		                              # time, beam_width] shape.
			sample_words = sample_words.transpose([2, 0, 1])
		return sample_words, infer_summary

	#######################  Important training private methods  ###########################

	def _compute_loss(self, logits):
	    """Compute optimization loss."""
	    target_output = self.iterator.target_output
	    
	    if self.time_major:
	    	target_output = tf.transpose(target_output)

	    max_time = self.get_max_time(target_output)
	    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
	        labels=target_output, logits=logits)
	    target_weights = tf.sequence_mask(
	        self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
	    
	    if self.time_major:
	    	target_weights = tf.transpose(target_weights)

	    loss = tf.reduce_sum(
	        crossent * target_weights) / tf.to_float(self.batch_size)

	    return loss

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
				    hparams, num_layers)

				encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
				    cell,
				    encoder_emb_inp,
				    dtype=dtype,
				    sequence_length=iterator.source_sequence_length,
				    time_major=self.time_major,
				    swap_memory=True)
	      
	    return encoder_outputs, encoder_state

	def _build_encoder_cell(self, hparams, num_layers,
	                          base_gpu=0):
	    """Build a multi-layer RNN cell that can be used by encoder."""

	    return model_helper.create_rnn_cell(
	        unit_type=hparams.unit_type,
	        num_units=hparams.num_units,
	        num_layers=num_layers,
	        num_residual_layers=0,
	        forget_bias=hparams.forget_bias,
	        dropout=hparams.dropout,
	        num_gpus=hparams.num_gpus,
	        mode=self.mode,
	        base_gpu=base_gpu,
	        single_cell_fn=self.single_cell_fn)

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
    		cell, decoder_initial_state = self._build_decoder_cell(
	          hparams, encoder_outputs, encoder_state,
	          iterator.source_sequence_length)

    		## Train or eval
      		if self.mode != tf.contrib.learn.ModeKeys.INFER:
      			# decoder_emp_inp: [max_time, batch_size, num_units]
		        target_input = iterator.target_input
		        if self.time_major:
		        	target_input = tf.transpose(target_input)

		        decoder_emb_inp = tf.nn.embedding_lookup(
		            self.embedding_decoder, target_input)

		        # Helper
				helper = tf.contrib.seq2seq.TrainingHelper(
				    decoder_emb_inp, iterator.target_sequence_length,
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
						helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
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

	    cell = tf.contrib.rnn.DeviceWrapper(cell,
	                                        model_helper.get_device_str(
	                                            num_layers - 1, self.num_gpus))

	    if hparams.pass_hidden_state:
			decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
			  cell_state=encoder_state)
	    else:
	    	decoder_initial_state = cell.zero_state(batch_size, dtype)

	    return cell, decoder_initial_state

	##############################  Helper methods  ###############################

	def init_embeddings(self, hparams, scope):
	    """Init embeddings."""
	    self.embedding_encoder, self.embedding_decoder = (
	        model_helper.create_emb_for_encoder_and_decoder(
	            share_vocab=hparams.share_vocab,
	            src_vocab_size=self.src_vocab_size,
	            tgt_vocab_size=self.tgt_vocab_size,
	            src_embed_size=hparams.num_units,
	            tgt_embed_size=hparams.num_units,
	            num_partitions=hparams.num_embeddings_partitions,
	            src_vocab_file=hparams.src_vocab_file,
	            tgt_vocab_file=hparams.tgt_vocab_file,
	            src_embed_file=hparams.src_embed_file,
	            tgt_embed_file=hparams.tgt_embed_file,
	            scope=scope,))

	def _get_learning_rate_warmup(self, hparams):
	    """Get learning rate warmup."""
	    warmup_steps = hparams.warmup_steps
	    warmup_scheme = hparams.warmup_scheme
	    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
	                    (hparams.learning_rate, warmup_steps, warmup_scheme))

	    # Apply inverse decay if global steps less than warmup steps.
	    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
	    # When step < warmup_steps,
	    #   learing_rate *= warmup_factor ** (warmup_steps - step)
	    if warmup_scheme == "t2t":
			# 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
			warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
			inv_decay = warmup_factor**(
			  tf.to_float(warmup_steps - self.global_step))
	    else:
			raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

	    return tf.cond(
	        self.global_step < hparams.warmup_steps,
	        lambda: inv_decay * self.learning_rate,
	        lambda: self.learning_rate,
	        name="learning_rate_warump_cond")

	def _get_learning_rate_decay(self, hparams):
		"""Get learning rate decay."""
		if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
			decay_factor = 0.5
			if hparams.decay_scheme == "luong5":
				start_decay_step = int(hparams.num_train_steps / 2)
				decay_times = 5
			elif hparams.decay_scheme == "luong10":
				start_decay_step = int(hparams.num_train_steps / 2)
				decay_times = 10
			elif hparams.decay_scheme == "luong234":
				start_decay_step = int(hparams.num_train_steps * 2 / 3)
				decay_times = 4
				remain_steps = hparams.num_train_steps - start_decay_step
				decay_steps = int(remain_steps / decay_times)
		elif not hparams.decay_scheme:  # no decay
			start_decay_step = hparams.num_train_steps
			decay_steps = 0
			decay_factor = 1.0
		elif hparams.decay_scheme:
			raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
		

		utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
		                "decay_factor %g" % (hparams.decay_scheme,
		                                     start_decay_step,
		                                     decay_steps,
		                                     decay_factor))

		return tf.cond(
		    self.global_step < start_decay_step,
		    lambda: self.learning_rate,
		    lambda: tf.train.exponential_decay(
		        self.learning_rate,
		        (self.global_step - start_decay_step),
		        decay_steps, decay_factor, staircase=True),
		    name="learning_rate_decay_cond")

if __init__ == '__main__':
	hparams = create_standard_hparams()

	DATA_DIR = 'data'
	src_file = os.path.join(DATA_DIR, 'instructions.txt')
	tgt_file = os.path.join(DATA_DIR, 'commands.txt')
	src_vocab_file = os.path.join(DATA_DIR,'instructions.vocab')
	tgt_vocab_file = os.path.join(DATA_DIR,'commands.vocab')

	graph = tf.Graph()

	with graph.as_default(), tf.container(scope or "train"):
		src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        	src_vocab_file, tgt_vocab_file, hparams.share_vocab)

		src_dataset = tf.data.TextLineDataset(src_file)
    	tgt_dataset = tf.data.TextLineDataset(tgt_file)

		# Create iterator
		iterator = iterator_utils.get_iterator(
	        src_dataset,
	        tgt_dataset,
	        src_vocab_table,
	        tgt_vocab_table,
	        hparams.batch_size,
	        sos=hparams.sos,
	        eos=hparams.eos,
	        random_seed=hparams.random_seed,
	        num_buckets=hparams.num_buckets,
	        src_max_len=hparams.src_max_len_infer,
	        tgt_max_len=hparams.tgt_max_len_infer,
	        skip_count=skip_count_placeholder)

		model = SimpleAttentionModel(
			hparams,
			iterator=iterator,
			mode=tf.contrib.learn.ModeKeys.TRAIN,
			source_vocab_table=src_vocab_table,
			target_vocab_table=tgt_vocab_table,
			scope=scope,
			extra_args=extra_args)

		train_sess = tf.Session()

		train_sess.run(tf.global_variables_initializer())
		# Because we run tf.contrib.lookup.index_table_from_file
		# in vocab_utils.create_vocab_tables
		# so this code will loads the vocabulary files, and initialize the values in the table
		train_sess.run(tf.tables_initializer())

		train_sess.run(iterator.initializer)

		while global_step < num_train_steps:
			try:
				step_result = model.train(train_sess)
				hparams.epoch_step += 1
			except tf.errors.OutOfRangeError:
				# Finished going through the training dataset.  Go to next epoch.
				hparams.epoch_step = 0

				utils.print_out(
			          "# Finished an epoch, step %d. Perform external evaluation" %
			          global_step)

				train_sess.run(
			          iterator.initializer,
			          feed_dict={model.skip_count_placeholder: 0})

				continue

		# Done training

		model.saver.save(
		      train_sess,
		      os.path.join("model", "translate.ckpt"),
		      global_step=global_step)