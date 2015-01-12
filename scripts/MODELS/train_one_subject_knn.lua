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
april_print_script_header(arg)
--
package.path = package.path .. ";./scripts/?.lua"
-- library loading and name import
local common = require "common"
local cmd    = require "cmd"
local posteriorKNN = knn.kdtree.posteriorKNN
---------------------------------------------------------------------------
-- command line options
local cmd_opt_parser = cmdOpt{
  program_name = arg[0]:basename(),
  argument_description = "",
  main_description = "Training of KNNs for Kaggle epilepsy challenge",
}
--
cmd.add_defopt(cmd_opt_parser)
cmd.add_cv(cmd_opt_parser)
cmd.add_knn(cmd_opt_parser)
cmd.add_help(cmd_opt_parser)
--
local params = cmd_opt_parser:parse_args(arg,"defopt")
---------------------------------------------------------------------------
-- extension of command line options
local log_scale = true
params.log_scale = log_scale
--
local SUBJECT = params.subject
local PREFIX = params.prefix or "./"
--
params.PREFIX = PREFIX
params.SUBJECT = SUBJECT
---------------------------------------------------------------------------
local rnd = random(params.seed)

print("# Loading training")
local all_train_data,list_names = common.load_data(params.fft, "*ictal*", params)
-- compute training standarization
local means,devs = common.compute_means_devs(all_train_data.input_dataset)
-- write standarization to output prefix
util.serialize({means,devs}, "%s/%s_standarization.lua"%{PREFIX,SUBJECT})
-- apply standarization to training data
all_train_data.input_dataset = common.apply_std(all_train_data.input_dataset,
                                                means, devs)
local input_size = all_train_data.input_dataset:patternSize()

print("# Train num patterns = ", all_train_data.input_dataset:numPatterns())
print("# Pattern size = ", all_train_data.input_dataset:patternSize())

-- KD-Tree training given a RNG and a dataset
local function train_knn(rnd, input_dataset)
  print("# Training KNN")
  local mat = input_dataset:toMatrix()
  assert((#mat:dim() == 2) and (mat:dim(2) == input_size))
  local kdt = knn.kdtree(mat:dim(2), rnd)
  kdt:push(mat)
  kdt:build()
  print("# Training done")
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

---------------------------
-- CROSS-VALIDATION LOOP --
---------------------------
local CV = common.train_with_crossvalidation
local AUC,MV,VAL_RESULTS = CV(list_names, all_train_data, params,
   -- train function
   function(train_data, val_data)
     local kdt = train_knn(rnd, train_data.input_dataset)
     return kdt, train_data.output_dataset
   end,
   -- classify function
   classify_matrix)
print("# LOSS:",table.unpack(MV))
print("# AUC:",table.unpack(AUC))
---------------------------------------------------------------------------

-- keep the training paths for this KNN
util.serialize(params, "%s/%s_params.lua"%{PREFIX,SUBJECT})

if params.test then
  local kdt = train_knn(rnd, all_train_data.input_dataset)
  print("# Loading test")
  local test_data,names = common.load_data(params.fft, common.TEST_MASK, params)
  test_data.input_dataset = common.apply_std(test_data.input_dataset,means,devs)
  print("# Test num patterns = ", test_data.input_dataset:numPatterns())
  local tr_out_ds = all_train_data.output_dataset
  common.save_test(params.test, names, test_data.input_dataset, params,
                   classify_matrix, kdt, tr_out_ds)
  return AUC[1],test_data.input_dataset:numPatterns()
else
  return AUC[1],all_train_data.input_dataset:numPatterns()
end
