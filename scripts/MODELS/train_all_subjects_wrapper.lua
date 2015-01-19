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
--
local script = table.remove(arg,1)
assert(not script:match("^%-%-"), "Needs the training script as first argument")
--
local auc_and_sizes = {}
print("#",script)
local train_script = assert(loadfile(script))
local ARG_SIZE = #arg
for _,subject in ipairs(common.SUBJECTS) do
  assert(#arg == ARG_SIZE)
  print("# TRAINING SUBJECT = ", subject)
  table.insert(arg, "--subject=%s"%{subject})
  print("# CMD = ", table.concat(arg, " "))
  print("####################################################################")
  auc_and_sizes[#auc_and_sizes+1] = table.pack( train_script(arg) )
  print("####################################################################")
  table.remove(arg,#arg)
  collectgarbage("collect")
end
-- show AUC for all subjects
iterator(ipairs(auc_and_sizes)):
  apply(function(k,v)
      print("# Subject " .. common.SUBJECTS[k] .. ":", v[1], v[2])
  end)
-- compute averaged AUC
local total_size = iterator(table.ivalues(auc_and_sizes)):field(2):reduce(math.add,0)
local averaged_auc = iterator(table.ivalues(auc_and_sizes)):
map(function(t) return t[1] * t[2]/total_size end):reduce(math.add,0)
print("# averaged AUC:",averaged_auc)
