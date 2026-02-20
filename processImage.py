import numpy as np
import time

def computeExpVal(oversam_pix_grid, xPower, yPower):
  """Writing a function that computes the expectation
     value for each component/term."""
  x, y = oversam_pix_grid
  eval_x = np.power(x, xPower)
  eval_y = np.power(y, yPower)
  return (np.mean(eval_x * eval_y))

def compute_pixel_weights(offsets, oversam = 6):
  x_array = np.linspace(-0.5 + 1 / (2 * oversam), 0.5 - 1 / (2 * oversam), oversam)
  y_array = np.linspace(-0.5 + 1 / (2 * oversam), 0.5 - 1 / (2 * oversam), oversam)
  # Locally define meshGrid for consistency with computeExpVal calls within this function
  meshGrid= np.meshgrid(x_array, y_array)
  sub_pixel_x, sub_pixel_y = meshGrid # x and y now refer to the (6,6) sub-pixel grid

  start = time.time()
  Ex2 = computeExpVal(meshGrid, 2, 0)
  Ey2 = computeExpVal(meshGrid, 0, 2)
  Exy = computeExpVal(meshGrid, 1, 1)
  Ex3 = computeExpVal(meshGrid, 3, 0)
  Ey3 = computeExpVal(meshGrid, 0, 3)
  Ex2y = computeExpVal(meshGrid, 2, 1)
  Exy2 = computeExpVal(meshGrid, 1, 2)
  Ex2y2 = computeExpVal(meshGrid, 2, 2)
  Ex3y = computeExpVal(meshGrid, 3, 1)
  Exy3 = computeExpVal(meshGrid, 1, 3)
  Ex4 = computeExpVal(meshGrid, 4, 0)
  Ey4 = computeExpVal(meshGrid, 0, 4)
  end = time.time()
  print("Time taken to compute expectation vals = ", (end-start))
  # The coefficient matrix built from the expectation values
  M = np.array([
      [1, 0, 0, Exy, Ex2, Ey2],
      [0, Ex2, Exy, Ex2y, Ex3, Exy2],
      [0, Exy, Ey2, Exy2, Ex2y, Ey3],
      [Exy, Ex2y, Exy2, Ex2y2, Ex3y, Exy3],
      [Ex2, Ex3, Ex2y, Ex3y, Ex4, Ex2y2],
      [Ey2, Exy2, Ey3, Exy3, Ex2y2, Ey4]
      ])

  # offsets is (4096, 4096, 6). We want to solve M * coeffs = deltas for each (4096, 4096) pixel.
  # Reshape offsets to (N, 6) where N = 4096 * 4096.
  offsets_flat = offsets.reshape(-1, 6) # Shape becomes (4096*4096, 6)

  # For np.linalg.solve(A, B) where A is (M,M) and B is (M,K)
  # we need to transpose offsets_flat to (6, 4096*4096)
  # so that M=6 and K=4096*4096
  start = time.time()
  coeff_flat_solved = np.linalg.solve(M, offsets_flat.T) # Result shape (6, 4096*4096)
  end = time.time()
  print("Time taken to solve equations = ", (end-start))

  # Transpose back and reshape to original (4096, 4096, 6) structure for coefficients
  coeffArray = coeff_flat_solved.T.reshape(offsets.shape[0], offsets.shape[1], 6) # Shape (4096, 4096, 6)

  start = time.time()
  
  #Build a basis of powers of subpixel x,y arrays to speed up computation 
  sub_pixel_basis = np.stack([
      np.ones_like(sub_pixel_x),
      sub_pixel_x,
      sub_pixel_y,
      sub_pixel_x * sub_pixel_y,
      sub_pixel_x**2,
      sub_pixel_y**2
  ], axis=-1) # Shape (oversam, oversam, 6) -> (6, 6, 6)

  # Use np.einsum for efficient calculation of the weighted sum.
  # 'ijk' refers to coeffArray (4096, 4096, 6) (pixel_row, pixel_col, coefficient_type)
  # 'lmk' refers to sub_pixel_basis (6, 6, 6) (sub_pixel_row, sub_pixel_col, coefficient_type)
  # The result 'ijlm' will be (4096, 4096, 6, 6) (pixel_row, pixel_col, sub_pixel_row, sub_pixel_col)
  weighted_sum_terms = np.einsum('ijk, lmk -> ijlm', coeffArray, sub_pixel_basis)

  # Add the constant term '1'
  weight_array = weighted_sum_terms

  end = time.time()
  print("Time to build weight array = ", (end-start))

  return weight_array # This will be (4096, 4096, 6, 6) as desired

def generateOffsetArray(offsets, imageSize = 4096, oversample = 6):
  #This function will copy a single array of offsets into 4096*4096*6 array to be used for testing.
  offsetArray = np.zeros((imageSize, imageSize, oversample))
  offsetArray[:imageSize, imageSize, :] = offsets
  return offsetArray


def processImage(oversampledImage, offsets, imageSize = 4096, oversample = 6):
  #oversampledImage is a (6*4096)^2 image
  #offsets refer to a (4096*4096*6) array that contains deltax, deltay etc for each pixel
  #want to return a single 4096*4096 image/array

  #First need to reshape image into (4096, 4096, 6, 6) 
  reshapedImage = oversampledImage.reshape(imageSize, oversample, imageSize, oversample).transpose(0, 2, 1, 3)

  weights = compute_pixel_weights(offsets, oversam = oversample)
  #should return a set of weights for each pixel (4088*4088*6*6)

  downsampledImage = np.sum(reshapedImage*weights, axis = (2,3))

  return downsampledImage

if __name__ == "__main__":
  #Tests the function with a simple test
  offsets = np.zeros((4096,4096,6))
  offsets[0,0,:] = np.array([1, 0, 0, 0, 1/12 * (1-1/36), 1/12 * (1-1/36)])
  print("Passing in offset array", offsets[0,0,:])
  blah = compute_pixel_weights(offsets)
  print("Computed Weights for the first pixel",blah[0,0,:,:])
