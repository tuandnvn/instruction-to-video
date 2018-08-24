import abc

import six

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.helper import _unstack_ta, GreedyEmbeddingHelper, Helper

class TrainingHelper(Helper):
    """A helper for use during training.  Only reads inputs.
    Returned sample_ids are the argmax of the RNN output logits.
    """

    def __init__(self, inputs, sequence_length, time_major=False, name=None):
        """Initializer.
        Args:
            inputs: A (structure of) input tensors.
            sequence_length: An int32 vector tensor.
            time_major: Python bool.  Whether the tensors in `inputs` are time major.
                If `False` (default), they are assumed to be batch major.
            name: Name scope for any created operations.
        Raises:
            ValueError: if `sequence_length` is not a 1D tensor.
        """
        with ops.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            self._inputs = inputs
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)

            # TFShouldUseWarningWrapper
            # Wrapper for objects that keeps track of their use.
            # Decorator that provides a warning if the wrapped object is never used.
            self._input_tas = nest.map_structure(_unstack_ta, inputs)

            print ('self._input_tas', self._input_tas.get_shape())
            self._sequence_length = ops.convert_to_tensor(
                    sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                        "Expected sequence_length to be a vector, but received shape: %s" %
                        self._sequence_length.get_shape())

            self._zero_inputs = nest.map_structure(
                    lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
            print ('self._zero_inputs', type(self._zero_inputs), self._zero_inputs.get_shape())

            self._batch_size = array_ops.size(sequence_length)

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        with ops.name_scope(name, "TrainingHelperInitialize"):
            finished = math_ops.equal(0, self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                    all_finished, lambda: self._zero_inputs,
                    lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
            return (finished, next_inputs)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with ops.name_scope(name, "TrainingHelperSample", [time, outputs]):
            sample_ids = math_ops.cast(
                    math_ops.argmax(outputs, axis=-1), dtypes.int32)
            return sample_ids

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """next_inputs_fn for TrainingHelper."""
        with ops.name_scope(name, "TrainingHelperNextInputs",
                                                [time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            def read_from_ta(inp):
                return inp.read(next_time)

            next_inputs = control_flow_ops.cond(
                    all_finished, lambda: self._zero_inputs,
                    lambda: nest.map_structure(read_from_ta, self._input_tas))
            return (finished, next_inputs, state)

def act_on_visual_input ( visual_input, pos, action_id ):
    """
    visual input: is an nxn array
    pos: a 2d position of the moving cell

    action_id : sampled from inference phase
    """
    size = int(np.sqrt(visual_size - 2))

    flatten_pos = visual_input[-2] * size + visual_input[-1]

    visual_input[flatten_pos]

    action = self._embedding_fn[action_id]

    def left () :
        return 

    def right () :
        return 

    def up () :
        return 

    def down () :
        return 

    def stop ():
        return visual_input

    return tf.case(
        {tf.equal(action, tf.constant('left')): left(), 
         tf.equal(action, tf.constant('right')): right(),
         tf.equal(action, tf.constant('up')): up(),
         tf.equal(action, tf.constant('down')): down()},
         default=stop(), exclusive=True)

class ControllerGreedyEmbeddingHelper(Helper):
    """A helper for use during inference.
    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token, 
                                        visual_input, visual_input_function):
        """Initializer.
        Args:
            embedding: A callable that takes a vector tensor of `ids` (argmax ids),
                or the `params` argument for `embedding_lookup`. The returned tensor
                will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
            visual_input: A numpy array of size = visual_size
                this visual_input will be a state of the helper and will be changed
                after each inference step
            visual_input_function: visual_input x pos x action-> visual_input
                (see act_on_visual_input for an implementation)
        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
                scalar.
        """
        self.visual_input = visual_input
        self.visual_input_function = visual_input_function

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                    lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
                start_tokens, dtype=dtypes.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(
                end_token, dtype=dtypes.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embedding_fn(self._start_tokens)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)
