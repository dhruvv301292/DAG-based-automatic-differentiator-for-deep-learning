from mytorch import tensor
import numpy as np

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))


def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K) where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    
    # INSTRUCTIONS
    # Find the sorted indices based on number of time steps in each sample
    # Extract slices from each sample and properly order them for the construction of the packed tensor. __getitem__ you defined for Tensor class will come in handy
    # Use the tensor.cat function to create a single tensor from the re-ordered segements
    # Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.
    indices = np.arange(0, len(sequence))[:, np.newaxis]
    lengths = np.array([i.shape[0] for i in sequence])[:, np.newaxis]
    combined = np.hstack((indices, lengths))

    combined[:, 1] = -combined[:, 1]
    combined = combined[combined[:, 1].argsort ()]
    combined[:, 1] = -combined[:, 1]
    sorted_indices = combined[:, 0]

    batch_sizes = []
    for i in range (combined[0, 1]):  # 0-5
        count = 0
        for j in sorted_indices:  # 0-2
            if i == 0 and count == 0:
                data = sequence[j][i]
                data = data.unsqueeze()
                count += 1
            else:
                if sequence[j].shape[0] - 1 >= i:
                    data = tensor.cat([data, sequence[j][i].unsqueeze()])
                    count += 1
        batch_sizes.append(count)

    return PackedSequence(data, sorted_indices, np.array(batch_sizes))

def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    
    # INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list (assuming the tensors were sorted in a descending fashion based on number of timesteps)
    # Construct these individual tensors using tensor.cat
    # Re-arrange this list of tensor based on ps.sorted_indices
    main_count = 0
    tlist = [None] * len(ps.sorted_indices)
    for i, batch in enumerate(ps.batch_sizes):
        t_count = batch
        while t_count != 0:
            if i == 0:
                tsr = ps.data[main_count].unsqueeze()
                tlist[ps.sorted_indices[batch - t_count]] = tsr
                t_count -= 1
                main_count += 1
            else:
                tlist[ps.sorted_indices[batch - t_count]] = tensor.cat([tlist[ps.sorted_indices[batch - t_count]], ps.data[main_count].unsqueeze()])
                t_count -= 1
                main_count += 1

    return tlist

