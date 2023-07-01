# -*- coding: utf-8 -*-

import time
import rasterio
import numpy as np
import starfm4py as stp
import matplotlib.pyplot as plt
from parameters import (path, sizeSlices)



start = time.time()

#Set the path where the images are stored
product = rasterio.open('Tests/Test_4/1-landsat.tif')
profile = product.profile
LandsatT0 = rasterio.open('Tests/Test_4/1-landsat.tif').read(1)
MODISt0 = rasterio.open('Tests/Test_4/1-modi0.tif').read(1)
MODISt1 = rasterio.open('Tests/Test_4/1-modi1.tif').read(1)
print("hey")
print(LandsatT0.shape[0])
print(LandsatT0.shape[1])

#TODO: teeno folders ki images ki shapes
#TODO: Test 4, Test 5 ki bhi shapes
#! Priority task

# # Set the path where to store the temporary results
# path_fineRes_t0 = 'Temporary/Tiles_fineRes_t0/'
# path_coarseRes_t0 = 'Temporary/Tiles_coarseRes_t0/'
# path_coarseRes_t1 = 'Temporary/Tiles_fcoarseRes_t1/'

# # Flatten and store the moving window patches
# fine_image_t0_par = stp.partition(LandsatT0, path_fineRes_t0)
# coarse_image_t0_par = stp.partition(MODISt0, path_coarseRes_t0)
# coarse_image_t1_par = stp.partition(MODISt1, path_coarseRes_t1)

# print ("Done partitioning!")

# # Stack the the moving window patches as dask arrays
# S2_t0 = stp.da_stack(path_fineRes_t0, LandsatT0.shape)
# S3_t0 = stp.da_stack(path_coarseRes_t0, MODISt0.shape)
# S3_t1 = stp.da_stack(path_coarseRes_t1, MODISt1.shape)

# shape = (sizeSlices, LandsatT0.shape[1])

# print ("Done stacking!")

# # Perform the prediction with STARFM
# for i in range(0, LandsatT0.size-sizeSlices*shape[1]+1, sizeSlices*shape[1]):
    
#     fine_image_t0 = S2_t0[i:i+sizeSlices*shape[1],]
#     coarse_image_t0 = S3_t0[i:i+sizeSlices*shape[1],]
#     coarse_image_t1 = S3_t1[i:i+sizeSlices*shape[1],]
#     prediction = stp.starfm(fine_image_t0, coarse_image_t0, coarse_image_t1, profile, shape)
    
#     if i == 0:
#         predictions = prediction
        
#     else:
#         predictions = np.append(predictions, prediction, axis=0)
  

# # Write the results to a .tif file   
# print ('Writing product...')
# profile = product.profile
# profile.update(dtype='float64', count=1) # number of bands
# file_name = path + 'prediction.tif'

# result = rasterio.open(file_name, 'w', **profile)
# result.write(predictions, 1)
# result.close()


# end = time.time()
# print ("Done in", (end - start)/60.0, "minutes!")

# # Display input and output
# plt.imshow(LandsatT0)                 # Isko comment karne se sab blank aata hai, image rtrieve nahi hoti probably
# plt.gray()                            # Isko comment karne se saara white chala gaya    
# plt.show()
# plt.imshow(MODISt0)
# plt.gray()                            # Ye command ek baar run hona bhi kaafi hai upar
# plt.show()
# plt.imshow(MODISt1)
# plt.gray()
# plt.show()	                          # Ye sirf latest retreived image ko show karti hai, do pat imshow lagaunga, fir do par show akrunga, to bhi last wali hi aayegi, ek baari hi
# plt.imshow(predictions)
# plt.gray()
# plt.show()
