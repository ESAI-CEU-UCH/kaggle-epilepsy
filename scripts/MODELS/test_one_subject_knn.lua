--[[
  This file is part of ESAI-CEU-UCH/kaggle-epilepsy (https://github.com/ESAI-CEU-UCH/kaggle-epilepsy)
  
  Copyright (c) 2014, ESAI, Universidad CEU Cardenal Herrera,
  (F. Zamora-Martínez, F. Muñoz-Malmaraz, P. Botella-Rocamora, J. Pardo)
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
]]
--
package.path = package.path .. ";./scripts/?.lua"
-- library loading and name import
local common = require "common"
local posteriorKNN = knn.kdtree.posteriorKNN
---------------------------------------------------------------------------
local model_path = assert(arg[1], "Needs a model path as first argument")
local subject = assert(arg[2], "Needs a subject name as second argument")
local test = assert(arg[3], "Needs an output test result filename as third argument")
io.open(test, "w"):close()
---------------------------------------------------------------------------
local params = loadfile("%s/%s_params.lua"%{model_path,subject})()
---------------------------------------------------------------------------
-- extension of command line options
local log_scale = true
params.log_scale = log_scale
---------------------------------------------------------------------------
local rnd = random(params.seed)
local all_train_data,list_names = common.load_data(params.fft, "*ictal*", params)
-- compute training standarization
local means,devs = common.compute_means_devs(all_train_data.input_dataset)
-- apply standarization to training data
all_train_data.input_dataset = common.apply_std(all_train_data.input_dataset,
                                                means, devs)
local input_size = all_train_data.input_dataset:patternSize()
------------------------------------------------
local function train_knn(rnd, input_dataset)
  local mat = input_dataset:toMatrix()
  assert((#mat:dim() == 2) and (mat:dim(2) == input_size))
  local kdt = knn.kdtree(mat:dim(2), rnd)
  kdt:push(mat)
  kdt:build()
  return kdt
end

-- looks for every row in pat and returns a matrix with preictal posteriors
local function classify_matrix(pat, ...)
  local kdt,out_ds = ...
  local function index(id) return out_ds:getPattern(id)[1] end
  local outputs = matrix(pat:dim(1), 1)
  for i=1,pat:dim(1) do
    local result = kdt:searchKNN(params.KNN, pat(i,':'))
    local p = posteriorKNN(result, index)
    outputs:set(i,1, p[1] or -99)
  end
  return outputs
end
------------------------------------------------
-- KD-Tree training given a RNG and a dataset
local kdt = train_knn(rnd, all_train_data.input_dataset)
--
print("# Loading test")
local test_data,names = common.load_data(params.fft, "*test*", params)
print("# Test num patterns = ", test_data.input_dataset:numPatterns())
assert(all_train_data.input_dataset:patternSize() == test_data.input_dataset:patternSize(),
       "Different input size between training and test data")
test_data.input_dataset = common.apply_std(test_data.input_dataset,means,devs)
local tr_out_ds = all_train_data.output_dataset
print("# Computing test output")
common.save_test(test, names, test_data.input_dataset, params,
                 classify_matrix, kdt, tr_out_ds)
