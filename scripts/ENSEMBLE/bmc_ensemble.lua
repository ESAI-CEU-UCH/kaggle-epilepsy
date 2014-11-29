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
package.path = package.path .. ";./scripts/?.lua"
-- library loading and name import
local common = require "common"
--
local rnd = random(12384)
local NaN = mathcore.limits.float.quiet_NaN()
local data = {}
local SAMPLES = 2000 --500
local SUBJECTS = common.SUBJECTS
-- the ensemble output is stored in this table
local result_test = { {'clip','preictal'} }
--
local function normalize(w)
  for _,row in matrix.ext.iterate(w,1) do row:abs():scal( 1/row:sum() ) end
end
--
local averaged_AUC = 0
local total = 0
--
for _,subject in ipairs(SUBJECTS) do
  collectgarbage("collect")
  local models = {}
  -- load each model outputs
  local N
  for i,path in ipairs(arg) do
    models[i] = {
      val  = matrix.fromTabFilename("%s/validation_%s.txt"%{path,subject}),
      test = iterator(io.open("%s/validation_%s.test.txt"%{path,subject}):
                        read("*a"):tokenize(" \t\n\r")):
      map(function(x) return x:tokenize(",") end):table()
    }
    N = N or models[i].val:dim(1)
    assert(N == models[i].val:dim(1))
  end
  --
  local in_ds_tbl = iterator(models):map(function(x) return x.val end):
  map(function(x) return dataset.matrix(x:select(2,1):clone()) end):table()
  local in_ds = dataset.join(in_ds_tbl)
  local out_ds = dataset.matrix(models[1].val:select(2,2):clone())
  -- AUTODIFF build a simple linear combination model
  local shared = { w1 = matrix(1, #models) }
  local log, T, M = autodiff.op.log, autodiff.op.transpose, autodiff.matrix
  local i,w1 = M("i w1")
  local f = log( i * T( w1 ) )
  -- build a neural network wrapper by using previous model
  local net = autodiff.ann.model(f, i, { w1 }, shared, #models, 1)
  -- LOSS
  local loss = ann.loss.cross_entropy()
  local bsize = N
  local trainer = trainable.supervised_trainer(net, loss, bsize)
  ---------------- BMC -----------------------------------------------------
  local w1 = shared.w1 -- weights of the model
  local wr = w1:clone() -- result weights
  local z = -math.huge
  local sum = 0
  trainer:build{ weights = { w1 = w1 } } -- build model with w1
  for _,w in trainer:iterate_weights() do w:zeros() end
  for i=1,SAMPLES do
    -- sample from uniform dirichlet
    for j=1,#models do w1[{1,j}] = -math.log(rnd:rand()) end
    normalize(w1)
    -- loss
    local loss = trainer:validate_dataset{ input_dataset = in_ds,
                                           output_dataset = out_ds }
    -- log-likelihood
    local loglh = -in_ds:numPatterns() * loss
    if loglh > z then -- for numerical stability
      wr:scal( math.exp(z - loglh) )
      z = loglh
    end
    local w = math.exp(loglh - z)
    wr[{}] = wr * sum / (sum + w) + w * w1
    sum = sum + w
  end
  normalize(wr)
  w1:copy(wr)
  ---------------------------------------------------------------------
  ---------------------------------------------------------------------
  ---------------------------------------------------------------------
  local val_input = in_ds:toMatrix()
  local val_output = trainer:calculate(val_input)
  local val_target = out_ds:toMatrix()
  --
  val_output:exp()
  local auc = metrics.roc(val_output, val_target):compute_area()
  fprintf(io.stderr,"# %20s  AUC=%.6f  %s\n", subject,auc,
          iterator.range(val_input:dim(2)):
            map(function(i)
                return ( metrics.roc(val_input(':',i), val_target):compute_area() )
            end):concat(" "))
  --
  local test_m_tbl = iterator(models):
    map(function(x)
        return matrix(#x.test,1,iterator(x.test):map(function(y) return y[2] end):table())
    end):table()
  local test_m = matrix.join(2, test_m_tbl)
  test_output = trainer:calculate(test_m)
  test_output:exp()
  for i=1,#models[1].test do
    table.insert(result_test, { models[1].test[i][1], test_output:get(i,1) })
  end
  --
  averaged_AUC = averaged_AUC + auc*test_output:size()
  total = total + test_output:size()
  fprintf(io.stderr, "#\t\t\t\t\t\t %s\n",
          table.concat(trainer:weights("w1"):toTable(), " "))
end
fprintf(io.stderr,"# AVERAGED AUC = %.6f\n", averaged_AUC/total)
print(iterator(result_test):map(table.unpack):concat(",","\n"))
