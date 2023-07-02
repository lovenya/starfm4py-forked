# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:33:49 2018

@author: Nikolina Mileva
"""

import zarr
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
from parameters import (windowSize, logWeight, temp, mid_idx, numberClass, spatImp, 
                        specUncertainty, tempUncertainty, path)

 


# Flatten blocks inside a dask array            
def block2row(array, row, folder, block_id=None):
    if array.shape[0] == windowSize:
        # Parameters	
        name_string = str(block_id[0] + 1)
        m,n = array.shape
        u = m + 1 - windowSize
        v = n + 1 - windowSize

    	# Get Starting block indices
        start_idx = np.arange(u)[:,None]*n + np.arange(v)

    	# Get offsetted indices across the height and width of input array
        offset_idx = np.arange(windowSize)[:,None]*n + np.arange(windowSize)

    	# Get all actual indices & index into input array for final output
        flat_array = np.take(array,start_idx.ravel()[:,None] + offset_idx.ravel())

        # Save to (dask) array in .zarr format
        file_name = path + folder + name_string + 'r' + row + '.zarr'
        zarr.save(file_name, flat_array)
    
    return array


# Divide an image in overlapping blocks   
def partition(image, folder):

    image_da = da.from_array(image, chunks = (windowSize,image.shape[1]))
    image_pad = da.pad(image_da, windowSize//2, mode='constant')

    print ("stp line 49 - def partition(image, _) - image's dimension: ") 
    print(image.shape)
    print ("stp line 51 - def partition - image_da's shape: ")
    print(image_da.shape)
    print ("stp line 53 - def partition - image_pad's shape: ")
    print(image_pad.shape)
    
    for i in range(0,windowSize):
        row = str(i)
        block_i = image_pad[i:,:]
        block_i_da = da.rechunk(block_i, chunks=(windowSize,image_pad.shape[1]))
        block_i_da.map_blocks(block2row, dtype=int, row=row, folder=folder).compute()


# Create a list of all files in the folder and stack them into one dask array
def da_stack(folder, shape):
    da_list = [] 
    full_path = path + folder
    max_blocks = shape[0]//windowSize + 1 
    
    for block in range(1,max_blocks + 1):
        for row in range(0,windowSize):
            name = str(block) + 'r' + str(row)
            full_name = full_path + name + '.zarr'
            try:
                da_array = da.from_zarr(full_name)
                da_list.append(da_array) 
            except Exception:
                continue
    da_stack_return =  da.rechunk(da.concatenate(da_list, axis=0), chunks = (shape[1],windowSize**2))

    print("stp line 80 - def da_stack - return da_stack_return's dimension: ")
    print(da_stack_return)

    return da_stack_return


# Calculate the spectral distance
def spectral_distance(fine_image_t0, coarse_image_t0):
    spec_diff = fine_image_t0 - coarse_image_t0
    spec_dist = 1/(abs(spec_diff) + 1.0)
    print ("Done spectral distance!", spec_dist)

    print ("stp line 92 - def spectral_distance(fine_image_t0, coarse_image_t0) - fine_image_t0's dimension: ")
    print (fine_image_t0.shape)
    print ("stp line 94 - def spectral_distance(fine_image_t0, coarse_image_t0) - coarse_image_t0's dimension: ")
    print (coarse_image_t0.shape)
    print ("stp line 96 - def spectral_distance - return spec_diff's dimension: ")
    print(spec_diff.shape)
    print ("stp line 98 - def spectral_distance - return spec_dist's dimension: ")
    print(spec_dist.shape)

    return spec_diff, spec_dist




# Calculate the temporal distance    
def temporal_distance(coarse_image_t0, coarse_image_t1):
    temp_diff = coarse_image_t1 - coarse_image_t0
    temp_dist = 1/(abs(temp_diff) + 1.0)
    print ("Done temporal distance!", temp_dist)

    print ("stp line 112 - def temporal_distance(coarse_image_t0, coarse_image_t1) - coarse_image_t0's dimension: ", coarse_image_t0.shape)
    print (coarse_image_t0.shape)
    print ("stp line 113 - def temporal_distance(coarse_image_t0, coarse_image_t1) - coarse_image_t1's dimension: " )
    print (coarse_image_t1.shape)
    print ("stp line 114 - def temporal_distance - return temp_diff's dimension: " )
    print (temp_diff.shape)
    print ("stp line 115 - def temporal_distance - return temp_dist's dimension: ")
    print (temp_dist.shape)

    return temp_diff, temp_dist
   

# Calculate the spatial distance    
def spatial_distance(array):
    coord          = np.sqrt((np.mgrid[0:windowSize,0:windowSize]-windowSize//2)**2)
    spat_dist      = np.sqrt(((0-coord[0])**2+(0-coord[1])**2))
    rel_spat_dist  = spat_dist/spatImp + 1.0 # relative spatial distance
    rev_spat_dist  = 1/rel_spat_dist # relative spatial distance reversed
    flat_spat_dist = np.ravel(rev_spat_dist)
    spat_dist_da   = da.from_array(flat_spat_dist, chunks=flat_spat_dist.shape)

    print ("stp line 129 - def spatial_distance - coord's dimension: ")
    print (coord.shape)
    print ("stp line 130 - def spatial_distance - spat_dist's dimension: ")
    print (spat_dist.shape)
    print ("stp line 131 - def spatial_distance - rel_spat_dist's dimension: ")
    print (rel_spat_dist.shape)
    print ("stp line 132 - def spatial_distance - rev_spat_dist's dimension: ")
    print (rev_spat_dist.shape)
    print ("stp line 133 - def spatial_distance - flat_spat_dist's dimension: ")
    print (flat_spat_dist.shape)
    print ("stp line 134 - def spatial_distance - return spat_dist_da's dimension: ")
    print (spat_dist_da.shape)

    print ("Done spatial distance!", spat_dist_da)
    
    return spat_dist_da


# Define the threshold used in the dynamic classification process
def similarity_threshold(fine_image_t0):#, st_dev):
    fine_image_t0 = da.where(fine_image_t0==0, np.nan, fine_image_t0)
    st_dev = da.nanstd(fine_image_t0, axis=1)# new
    sim_threshold = st_dev*2/numberClass 
    print ("Done similarity threshold!", sim_threshold)

    print("stp line 140 - def similarity_threshold(fine_image_t0) - fine_image_t0's dimension: ")
    print(fine_image_t0.shape)
    print("stp line 141 - def similarity_threshold - st_dev's dimension: ")
    print(st_dev.shape)
    print("stp line 142 - def similarity_threshold - return sim_threshold's dimension: ")
    print(sim_threshold.shape)

    return sim_threshold


# Define the spectrally similar pixels within a moving window    
def similarity_pixels(fine_image_t0):
    sim_threshold = similarity_threshold(fine_image_t0)
    # possible to implement as sparse matrix
    similar_pixels = da.where(abs(fine_image_t0 - 
                                  fine_image_t0[:,mid_idx][:,None])
        <= sim_threshold[:,None], 1, 0) #sim_threshold[:,mid_idx][:,None], 1, 0) # new
    print ("Done similarity pixels!", similar_pixels)
   
    print("stp line 156 - def similarity_pixels(fine_image_t0) - fine_image_t0's dimension: ")
    print(fine_image_t0.shape)
    print("stp line 157 - def similarity_pixels - sim_threshold's dimension: " )
    print(sim_threshold.shape)
    print("stp line 158 - def similarity_pixels - return similar_pixels's dimension: ")
    print(similar_pixels.shape)

    return similar_pixels
        

# Apply filtering on similar pixels 
def filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, temp_diff):
    similar_pixels = similarity_pixels(fine_image_t0) 
    max_spec_dist  = abs(spec_diff)[:,mid_idx][:,None] + specUncertainty + 1
    max_temp_dist  = abs(temp_diff)[:,mid_idx][:,None] + tempUncertainty + 1  
    spec_filter    = da.where(spec_dist>1.0/max_spec_dist, 1, 0)
    st_filter      = spec_filter
    
    if temp == True:
        temp_filter = da.where(temp_dist>1.0/max_temp_dist, 1, 0)
        st_filter = spec_filter*temp_filter  
        
    similar_pixels_filtered = similar_pixels*st_filter
    print ("Done filtering!", similar_pixels_filtered)

    print ("stp line 178 - def filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, temp_diff) - fine_image_t0's dimension: " )
    print(fine_image_t0.shape)
    print ("stp line 179 - def filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, temp_diff) - spec_dist's dimension: " )
    print(spec_dist.shape)
    print ("stp line 180 - def filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, temp_diff) - temp_dist's dimension: " )
    print(temp_dist.shape)
    print ("stp line 181 - def filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, temp_diff) - spec_diff's dimension: " )
    print(spec_diff.shape)
    print ("stp line 182 - def filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, temp_diff) - temp_diff's dimension: " )
    print(temp_diff.shape)
    
    print ("stp line 184 - def filtering - similar_pixels's dimension: ")
    print(similar_pixels.shape)
    print ("stp line 185 - def filtering - max_spec_dist's dimension: ")
    print(max_spec_dist.shape)
    print ("stp line 186 - def filtering - max_temp_dist's dimension: ")
    print(max_temp_dist.shape)
    print ("stp line 187 - def filtering - spec_filter's dimension: ")
    print(spec_filter.shape)
    print ("stp line 188 - def filtering - st_filter's dimension: ")
    print(st_filter.shape)
    print ("stp line 189 - def filtering - return similar_pixels_filtered's dimension: ")
    print(similar_pixels_filtered.shape)   


    return similar_pixels_filtered # sim_pixels_sparse
    

# Calculate the combined distance
def comb_distance(spec_dist, temp_dist, spat_dist):

    print ("stp line 198 - def comb_distance(spec_dist, temp_dist, spat_dist) - spec_dist's, passed as a argument in the function, dimension : ", spec_dist.shape)
    print ("stp line 199 - def comb_distance(spec_dist, temp_dist, spat_dist) - temp_dist's, passed as a argument in the function, dimension : ", temp_dist.shape)


    if logWeight == True:
        spec_dist = da.log(spec_dist + 1)
        temp_dist = da.log(temp_dist + 1)
    
    comb_dist = da.rechunk(spec_dist*temp_dist*spat_dist, 
                           chunks=spec_dist.chunksize)
    print ("Done comb distance!", comb_dist)
    
    print ("stp line 210 - def comb_distance - after ruunning in the for loop in the function, spec_dist's dimension : " + spec_dist.shape)
    print ("stp line 211 - def comb_distance - after ruunning in the for loop in the function, temp_dist's dimension : " + temp_dist.shape)
    print ("stp line 212 - def comb_distance - return comb_dist's dimension: " + comb_dist.shape)

    return comb_dist
    
        
# Calculate weights
def weighting(spec_dist, temp_dist, comb_dist, similar_pixels_filtered):
    # Assign max weight (1) when the temporal or spectral distance is zero
    zero_spec_dist = da.where(spec_dist[:,mid_idx][:,None] == 1, 1, 0)
    zero_temp_dist = da.where(temp_dist[:,mid_idx][:,None] == 1, 1, 0)
    zero_dist_mid  = da.where((zero_spec_dist == 1), 
                             zero_spec_dist, zero_temp_dist)
    shape          = da.subtract(spec_dist.shape,(0,1))
    zero_dist      = da.zeros(shape, chunks=(spec_dist.shape[0],shape[1]))
    zero_dist      = da.insert(zero_dist, [mid_idx], zero_dist_mid, axis=1)
    weights        = da.where((da.sum(zero_dist,1)[:,None] == 1), zero_dist, comb_dist)
    
    # Calculate weights only for the filtered spectrally similar pixels
    weights_filt   = weights*similar_pixels_filtered
    
    # Normalize weights
    norm_weights   = da.rechunk(weights_filt/(da.sum(weights_filt,1)[:,None]), 
                              chunks = spec_dist.chunksize)
    
    print ("Done weighting!", norm_weights)

    print ("stp line 238 - def weighting(spec_dist, temp_dist, comb_dist, similar_pixels_filtered) - spec_dist's dimension: " + spec_dist.shape)
    print ("stp line 239 - def weighting(spec_dist, temp_dist, comb_dist, similar_pixels_filtered) - temp_dist's dimension: " + temp_dist.shape)
    print ("stp line 240 - def weighting(spec_dist, temp_dist, comb_dist, similar_pixels_filtered) - comb_dist's dimension: " + comb_dist.shape)
    print ("stp line 241 - def weighting(spec_dist, temp_dist, comb_dist, similar_pixels_filtered) - similar_pixels_filtered's dimension: " + similar_pixels_filtered.shape)
    
    print ("stp line 243 - def weighting - zero_spec_dist's dimension: " + zero_spec_dist.shape)
    print ("stp line 244 - def weighting - zero_temp_dist's dimension: " + zero_temp_dist.shape)
    print ("stp line 245 - def weighting - zero_dist_mid's dimension: " + zero_dist_mid.shape)
    print ("stp line 246 - def weighting - shape's dimension: " + shape.shape)
    print ("stp line 247 - def weighting - zero_dist's dimension: " + zero_dist.shape)
    print ("stp line 248 - def weighting - zero_dist's dimension: " + zero_dist.shape)
    print ("stp line 249 - def weighting - weights's dimension: " + weights.shape)
    print ("stp line 250 - def weighting - weights_filt's dimension: " + weights_filt.shape)
    print ("stp line 251 - def weighting - return norm_weights's dimension: " + norm_weights.shape)
    
    return norm_weights


# Derive fine resolution reflectance for the day of prediction 
def predict(fine_image_t0, coarse_image_t0, coarse_image_t1, shape):

    print ("stp line 259 - def predict(fine_image_t0, coarse_image_t0, coarse_image_t1, shape) - fine_image_t0  's dimension: ", fine_image_t0.shape)
    print ("stp line 260 - def predict(fine_image_t0, coarse_image_t0, coarse_image_t1, shape) - coarse_image_t0's dimension: " + coarse_image_t0.shape)
    print ("stp line 261 - def predict(fine_image_t0, coarse_image_t0, coarse_image_t1, shape) - coarse_image_t1's dimension: " + coarse_image_t1.shape)
    print ("stp line 262 - def predict(fine_image_t0, coarse_image_t0, coarse_image_t1, shape) - shape's dimension: " + shape.shape)

    spec      = spectral_distance(fine_image_t0, coarse_image_t0)
    spec_diff = spec[0]
    spec_dist = spec[1]
    temp      = temporal_distance(coarse_image_t0, coarse_image_t1)
    temp_diff = temp[0] 
    temp_dist = temp[1]
    spat_dist = spatial_distance(fine_image_t0)
    comb_dist = comb_distance(spec_dist, temp_dist, spat_dist)
    similar_pixels = filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, 
                               temp_diff)
    weights            = weighting(spec_dist, temp_dist, comb_dist, similar_pixels)    
    pred_refl          = fine_image_t0 + temp_diff
    weighted_pred_refl = da.sum(pred_refl*weights, axis=1)   
    prediction         = da.reshape(weighted_pred_refl, shape)
    print ("Done prediction!")
    
    print ("stp line 280 - def predict - spec's dimension: " + spec.shape)
    print ("stp line 281 - def predict - spec_diff's dimension: " + spec_diff.shape)
    print ("stp line 282 - def predict - spec_dist's dimension: " + spec_dist.shape)
    print ("stp line 283 - def predict - temp's dimension: " + temp.shape)
    print ("stp line 284 - def predict - temp_diff's dimension: " + temp_diff.shape)
    print ("stp line 285 - def predict - temp_dist's dimension: " + temp_dist.shape)
    print ("stp line 286 - def predict - spat_dist's dimension: " + spat_dist.shape)
    print ("stp line 287 - def predict - comb_dist's dimension: " + comb_dist.shape)
    print ("stp line 288 - def predict - similar_pixels's dimension: " + similar_pixels.shape)
    print ("stp line 289 - def predict - weights's dimension: " + weights.shape)
    print ("stp line 290 - def predict - pred_refl's dimension: " + pred_refl.shape)
    print ("stp line 291 - def predict - weighted_pred_refl's dimension: " + weighted_pred_refl.shape)
    print ("stp line 292 - def predict - return prediction's dimension: " + prediction.shape)

    return prediction
    
 
# Compute the results (converts the dask array to a numpy array)   
def starfm(fine_image_t0, coarse_image_t0, coarse_image_t1, profile, shape):
    print ('Processing...')
    prediction_da = predict(fine_image_t0, coarse_image_t0, coarse_image_t1, shape)
    with ProgressBar():
         prediction = prediction_da.compute()
    
    return prediction