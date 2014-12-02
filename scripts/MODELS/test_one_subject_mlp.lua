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
--
-- library loading and name import
local common = require "common"
---------------------------------------------------------------------------
local model_path = assert(arg[1], "Needs a model path as first argument")
local subject = assert(arg[2], "Needs a subject name as second argument")
local test = assert(arg[3], "Needs an output test result filename as third argument")
local FFT = arg[4]
local COR = arg[5]
io.open(test, "w"):close()
---------------------------------------------------------------------------
local model_table = loadfile("%s/%s.net"%{model_path,subject})()
local means,devs = table.unpack( loadfile("%s/%s_standarization.lua"%
                                            {model_path,subject})() )
--
local models = model_table.models
local params = model_table.params
if FFT then params.fft = FFT end
if COR then params.cor = COR end
---------------------------------------------------------------------------
assert(#models > 0, "None model has been trained :S")
---------------------------------------------------------------------------
-- perform test prediction if given test output filename
print("# Loading test")
local test_data,names = common.load_data(params.fft, "*test*", params)
--
test_data.input_dataset = common.apply_std(test_data.input_dataset,means,devs)
print("# Test num patterns = ", test_data.input_dataset:numPatterns())
print("# Computing test output")
common.save_test(test, names, test_data.input_dataset,
                 params,
                 function(pat, ...)
                   local models = ...
                   local result = matrix(pat:dim(1),1):zeros()
                   for _,m in ipairs(models) do
                     local out = m:use_dataset{ input_dataset = dataset.matrix(pat) }:toMatrix()
                     if log_scale then out:exp() end
                     result:axpy(1.0, out)
                   end
                   result:scal(1/#models)
                   if log_scale then result:log() end
                   return result
                 end,
                 models)
