"""
Author: eranroz

Find the most probable path through the model

Dynamic programming algorithm for decoding the states.
Implementation according to Durbin, Biological sequence analysis [p. 57]
"""
import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float
ctypedef np.float_t DTYPE_t
INT_TYPE = np.int
ctypedef np.int_t DTYPE_int

@cython.boundscheck(False)
@cython.profile(False)
@cython.wraparound(False)
def viterbi(np.ndarray[DTYPE_t, ndim=2] emission_seq not None, np.ndarray[DTYPE_t, ndim=2] state_transition not None):
	state_transition[0, 0] = 1
	return _viterbi(emission_seq, np.log(state_transition.T))

@cython.boundscheck(False)
@cython.profile(False)
@cython.wraparound(False)
cdef max_argmax(np.ndarray[DTYPE_t, ndim=2] arr, np.ndarray[DTYPE_int, ndim=1] argmaxes, np.ndarray[DTYPE_t, ndim=1] maxes):
	cdef int i, j
	for i in range(0, arr.shape[0]):
		argmaxes[i] = 0
		maxes[i] = arr[i,0]
		for j in range(1, arr.shape[1]):
			if maxes[i]<arr[i,j]:
				argmaxes[i], maxes[i] = j, arr[i,j]

@cython.boundscheck(False)
@cython.profile(False)
@cython.infer_types(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_int, ndim=1] _viterbi(np.ndarray[DTYPE_t, ndim=2] emission_seq, np.ndarray[DTYPE_t, ndim=2] l_state_trans_mat_T):
	cdef np.ndarray[DTYPE_int, ndim=2] ptr_mat = np.zeros((emission_seq.shape[0], emission_seq.shape[1]), dtype=INT_TYPE)
	cdef np.ndarray[DTYPE_int, ndim=1] most_probable_path = np.zeros(emission_seq.shape[0], dtype=INT_TYPE)	

	emission_iterator = iter(emission_seq)
	ptr_iterator = iter(ptr_mat)
	#intial condition is begin state	
	prev = next(emission_iterator) + 1+l_state_trans_mat_T[1:, 0]
	
	next(ptr_iterator)[...] = np.argmax(prev)
	cdef int end_state = 0  # termination step
	end_transition = l_state_trans_mat_T[end_state, 1:]
	l_state_trans_mat_T = l_state_trans_mat_T[1:, 1:].copy(order='F')
	
	#recursion step
	cdef int t
	for t in range(0,emission_seq.shape[0]-1):
		#for emission_symbol in emission_iterator:
		max_argmax(prev+l_state_trans_mat_T, next(ptr_iterator), prev)
		prev+=next(emission_iterator)

	most_probable_path[emission_seq.shape[0]-1] = np.argmax(prev + end_transition)
	cdef int i
	for i in range(emission_seq.shape[0]-1, 0, -1):
		most_probable_path[i - 1] = ptr_mat[i, most_probable_path[i]]
	
	return most_probable_path

class BackwardForwardResult:
	def __init__(self, log_p_model, forward, backward, s_j):
		self.model_p = log_p_model
		self.state_p = backward * forward
		self.forward = forward
		self.backward = backward
		self.scales = s_j
		

#def forward_backward(np.ndarray[DTYPE_t, ndim=2] state_trans_mat, np.ndarray[DTYPE_t, ndim=2, mode='c'] emission_seq, model_end_state=False):
@cython.boundscheck(False)
@cython.profile(False)
@cython.wraparound(False)
@cython.cdivision(True)
def forward_backward(state_trans_mat, emission_seq, model_end_state=False):
	"""
	Calculates the probability for the model and each step in it

	@param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
	@param model: an HMM model to calculate on the given symbol sequence
	@param model_end_state: whether to consider end state or not

	Remarks:
	this implementation uses scaling variant to overcome floating points errors.
	"""
	cdef int n_states = state_trans_mat.shape[0]
	
	#cdef np.ndarray[DTYPE_t, ndim=2, mode='fortran'] forward = np.zeros((emission_seq.shape[0], n_states - 1), order='F')	# minus the begin state
	#cdef np.ndarray[DTYPE_t, ndim=2, mode='fortran'] backward = np.zeros((emission_seq.shape[0], n_states - 1), order='F')
	#cdef np.ndarray[DTYPE_t, ndim=2] prev_back
	forward_iterator=np.nditer([emission_seq, None, None], 
             flags=['external_loop','reduce_ok'], 
             op_flags=[['readonly'], 
                       ['readwrite','allocate', 'no_broadcast'], 
                       ['readwrite','allocate', 'no_broadcast']],
             op_axes=[[-1,0,1], [-1,0,1], [-1,0,-1]])
			 
	#-----	  forward algorithm	  -----
	#intial condition is begin state (in Durbin there is another forward - the begin = 1)
	#cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] emission_i
	#cdef np.ndarray[DTYPE_t, ndim=1,mode='c'] forward_i
	#cdef np.ndarray[DTYPE_t, ndim=1] scaling_i
	
	tup = next(forward_iterator) #emission_i, forward_i,scaling_i
	tup[1][...] = state_trans_mat[0, 1:] * tup[0]
	tup[2][...]=np.sum(tup[1])
	tup[1][...] /= tup[2]
	prev_forward = tup[1]
	#recursion step
	
	real_transitions_T = state_trans_mat[1:, 1:].T.copy(order='C')
	real_transitions_T2 = state_trans_mat[1:, 1:].copy(order='C')
	#p_state_transition = np.zeros(n_states - 1)
	summing_arr = np.ones(n_states - 1)
	dot=np.dot
	for tup in forward_iterator:#emission_i, forward_i,scaling_i
		prev_forward = tup[0]*dot(real_transitions_T, prev_forward)

		# scaling - see Rabiner p. 16, or Durbin p. 79
		scaling=tup[2]
		scaling[...] = dot(summing_arr, prev_forward)	 # dot is actually faster then np.sum(prev_forward)
		tup[1][...] = prev_forward = prev_forward / scaling

	forward = forward_iterator.operands[1]
	s_j = forward_iterator.operands[2]
	#end transition
	log_p_model = np.sum(np.log(s_j))
	if model_end_state:	 # Durbin - with end state
		end_state = 0  # termination step
		end_transition = forward[emission_seq.shape[0] - 1, :] * state_trans_mat[1:, end_state]
		log_p_model += np.log(sum(end_transition))

	#-----	backward algorithm	-----
	#intial condition is end state
	if model_end_state:
		prev_back = (state_trans_mat[1:, 0])	 # Durbin p.60
	else:
		prev_back = np.ones(n_states-1)  # Rabiner p.7 (24)

	backward_iterator=np.nditer([emission_seq[:0:-1], s_j[:0:-1], None], 
             flags=['external_loop','reduce_ok'], 
             op_flags=[['readonly'], 
                       ['readonly'], 
                       ['readwrite','allocate', 'no_broadcast']],
             op_axes=[[-1,0,1], [-1,0,-1], [-1,0,1]])
	
	

	#recursion step
	for tup in backward_iterator:#emission_i, scale, backward_i
		tup[2][...] = prev_back = dot(real_transitions_T2, prev_back * tup[0]) / tup[1]
	
	if model_end_state:
		backward=np.append(backward_iterator.operands[2][::-1],state_trans_mat[1:, 0][None,:],axis=0)	 # Durbin p.60
	else:
		backward=np.append(backward_iterator.operands[2][::-1],np.ones((1, n_states-1)),axis=0)  # Rabiner p.7 (24)

	bf_result = BackwardForwardResult(log_p_model, forward, backward, s_j)
	return bf_result
