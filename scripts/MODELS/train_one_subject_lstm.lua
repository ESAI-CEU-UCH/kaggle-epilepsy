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
--
-- library loading and name import
local common = require "common"
local cmd    = require "cmd"
--
local cmd_opt_parser = cmdOpt{
  program_name = arg[0]:basename(),
  argument_description = "",
  main_description = "Training of MLPs for Kaggle epilepsy challenge",
}
--
cmd.add_defopt(cmd_opt_parser)
cmd.add_cv(cmd_opt_parser)
cmd.add_lstm(cmd_opt_parser)
cmd.add_help(cmd_opt_parser)
--
local params = cmd_opt_parser:parse_args(arg,"defopt")
---------------------------------------------------------------------------
-- random number generators
local weights_random = random(params.wseed)
local shuffle_random = random(params.sseed)
local perturbation_random = random(params.pseed)
local perturbation_random_2 = random(params.pseed2)
local perturbation_random_3 = random(params.pseed3)
local wnoise = (params.wnoise>0.0) and stats.dist.normal(0,params.wnoise) or nil
-- loss function data
local log_scale,smooth,loss_param = common.loss_stuff(params.loss)
local loss_function = ann.loss[params.loss](loss_param)
-- basic options for subject and output prefix
local SUBJECT = params.subject
local PREFIX = params.prefix or "./"
-- extension of command line options
if params.bsize == 0 then params.bsize = nil end
params.log_scale = log_scale
params.PREFIX = PREFIX
params.SUBJECT = SUBJECT
--
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
-- number of rows in every filename
local NROWS = all_train_data.input_dataset:numPatterns() / #list_names
assert(math.floor(NROWS) == NROWS)
--
print("# Train num patterns = ", all_train_data.input_dataset:numPatterns())
print("# Pattern size = ", all_train_data.input_dataset:patternSize())
print("# Input size = ", input_size)
--
-- build the ANN
local isize = input_size
local the_net = ann.graph('seizure_predictor')
local aux = 'input'
if params.h0 > 0 then
  aux = the_net:connect( aux,
                         ann.components.hyperplane{ input=isize, output=params.h0 },
                         ann.components.actf[params.actf]() )
  if params.dropout > 0.0 then
    aux = the_net:connect(aux, ann.components.dropout{ prob=params.dropout })
  end
  isize = params.h0
end
aux = the_net:connect( aux,
                       ann.graph.blocks.lstm{ name="LSTM",
                                              input=isize,
                                              output=params.lstm,
                                              actf=params.actf } )
if params.dropout > 0.0 then
  aux = the_net:connect(aux, ann.components.dropout{ prob=params.dropout })
end
isize = params.lstm
for hsize in iterator{ params.h1, params.h2, params.h3,
                       params.h4, params.h5, params.h6 } do
  if hsize > 0 then
    aux = the_net:connect( aux,
                           ann.components.hyperplane{ input=isize, output=hsize },
                           ann.components.actf[params.actf]() )
    if params.dropout > 0.0 then
      aux = the_net:connect(aux, ann.components.dropout{ prob=params.dropout })
    end
    isize = hsize
  end
end

aux = the_net:connect( aux, ann.components.hyperplane{ input=isize,
                                                       output=1,
                                                       dot_product_weights="wN",
                                                       bias_weights="bN" } )
if params.log_scale then
  aux = the_net:connect( aux, ann.components.actf.log_logistic() )
else
  aux = the_net:connect( aux, ann.components.actf.logistic() )
end
the_net:connect( aux, 'output' )
collectgarbage("collect")

local to_dataset_token = function(ds)
  return class.is_a(ds,dataset) and dataset.token.wrapper(ds) or ds
end

local function forward(the_net, seq, during_training, it)
  the_net:reset(it)
  local rewrapped_slice, output
  for i,slice in matrix.ext.iterate(seq, 1) do
    output = the_net:forward(slice, during_training)
  end
  return output
end

-- TRAINING SEQUENCE FUNCTION --
local function train_sequence(trainer, in_seq, out_seq)
  local weights = trainer:get_weights_table()
  if params.wnoise > 0.0 then
    for _,w in trainer:iterate_weights() do
      local p = wnoise:sample(perturbation_random_3, w:size())
      w:axpy(1.0, p)
    end
  end
  local the_net = trainer:get_component()
  local loss = trainer:get_loss_function()
  local opt = trainer:get_optimizer()
  local l,grads = opt:execute(function(x,it)
      if x~=weights then
        trainer:build{ weights=x }
        weights=x
      end
      local y_hat = forward(the_net, in_seq, true, it)
      local y = out_seq
      loss:accum_loss( loss:compute_loss(y_hat, y) )
      the_net:backprop( loss:gradient(y_hat, y) )
      local grads = the_net:compute_gradients()
      if smooth then
        matrix.dict.scal(grads, 1/math.sqrt(in_seq:dim(2)))
      end
      return loss:get_accum_loss(), grads
                              end,
    weights)
end

-- TRAINING DATASET FUNCTION --
local function train_dataset(trainer, train_data, bsize, NROWS)
  local in_ds = to_dataset_token(train_data.input_dataset)
  local out_ds = to_dataset_token(train_data.output_dataset)
  local shuffle = train_data.shuffle
  local N = in_ds:numPatterns() / NROWS
  assert(in_ds:numPatterns() % NROWS == 0)
  local loss = trainer:get_loss_function()
  loss:reset()
  for i=1,train_data.replacement,bsize do
    local in_seqs, out_seqs = {}, {}
    for j=1,bsize do
      local ipat = shuffle:randInt(0,N-1) * NROWS
      local start,stop = ipat+1, ipat+NROWS
      local idxs = iterator.range(start,stop):table()
      in_seqs[j] = in_ds:getPatternBunch(idxs)
      table.insert(out_seqs, stop)
      local aux_out = out_ds:getPatternBunch(idxs)
      local sum = aux_out:sum()
      assert(sum == 0 or sum == NROWS)
    end
    local input = matrix(NROWS, #in_seqs, in_ds:patternSize())
    for j,slice in matrix.ext.iterate(input, 2) do slice:copy( in_seqs[j] ) end
    local target = out_ds:getPatternBunch(out_seqs)
    train_sequence( trainer, input, target )
  end
  return loss:get_accum_loss()
end

-- VALIDATE DATASET FUNCTION --
local function validate_dataset(trainer, val_data, bsize, NROWS)
  local in_ds = to_dataset_token(val_data.input_dataset)
  local out_ds = to_dataset_token(val_data.output_dataset)
  local N = in_ds:numPatterns() / NROWS
  assert(in_ds:numPatterns() % NROWS == 0)
  local the_net = trainer:get_component()
  local loss = trainer:get_loss_function()
  loss:reset()
  for i=1,N,bsize do
    local in_seqs, out_seqs = {}, {}
    for k=i,math.min(N,i+bsize-1) do
      local ipat = (k-1) * NROWS
      local start,stop = ipat+1, ipat+NROWS
      local idxs = iterator.range(start,stop):table()
      in_seqs[#in_seqs+1] = in_ds:getPatternBunch(idxs)
      table.insert(out_seqs, stop)
      local aux_out = out_ds:getPatternBunch(idxs)
      local sum = aux_out:sum()
      assert(sum == 0 or sum == NROWS)
    end
    local input = matrix(NROWS, #in_seqs, in_ds:patternSize())
    for j,slice in matrix.ext.iterate(input, 2) do slice:copy( in_seqs[j] ) end
    local target = out_ds:getPatternBunch(out_seqs)
    local y_hat = forward(the_net, input)
    local y = target
    loss:accum_loss( loss:compute_loss(y_hat, y) )
  end
  return loss:get_accum_loss()
end

-- CLASSIFY FUNCTION --
local function classify(pat,...)
  assert(pat:dim(1) % NROWS == 0)
  local N = pat:dim(1) / NROWS
  pat = pat:contiguous():rewrap(N, NROWS, pat:dim(2)):t(1,2):clone()
  local trainer = ...
  local the_net = trainer:get_component()
  local y_hat = forward(the_net, pat)
  return y_hat
end

---------------------------
-- CROSS-VALIDATION LOOP --
---------------------------
local models = {}
local CV = common.train_with_crossvalidation
local AUC,MV,VAL_RESULTS = CV(list_names, all_train_data, params,
   -- train function in CV algorithm
   function(train_data, val_data)
     local non_perturbed_input_ds = train_data.input_dataset
     local bsize = params.bsize or train_data.input_dataset:numPatterns()
     print("# BSIZE = ", bsize)
     train_data.shuffle       = shuffle_random
     train_data.replacement   = bsize
     train_data.input_dataset =
       dataset.token.filter( train_data.input_dataset,
                             ann.components.gaussian_noise{ random=perturbation_random,
                                                            var = params.var } )     
     collectgarbage("collect")
     --
     local trainer = trainable.supervised_trainer(the_net:clone(),
                                                  loss_function,
                                                  bsize,
                                                  ann.optimizer.adadelta(),
                                                  smooth)
     local _,w = trainer:build{ input = input_size, output = 1 }
     print("# MODEL SIZE = ", trainer:size())
     print("# RANDOMIZE WEIGHTS")
     trainer:randomize_weights{ inf    = -params.wrange,
                                sup    =  params.wrange,
                                random =  weights_random,
                                use_fanin  = true,
                                use_fanout = true }
     local bias_mask = "^.*b[0-9]*$"
     if params.actf == "relu" then
       for _,b in trainer:iterate_weights(bias_mask) do b:fill(0.1) end
     end
     -- set learning hyper-parameters
     if params.mnp then
       trainer:set_option("max_norm_penalty", params.mnp)
       trainer:set_layerwise_option(bias_mask, "max_norm_penalty", 0.0)
     end
     if params.wd then
       trainer:set_option("weight_decay", params.wd)
       trainer:set_layerwise_option(bias_mask, "weight_decay", 0.0)
     end
     -- pocket algorithm object
     local pocket = trainable.train_holdout_validation{
       stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(1.5),
       min_epochs = params.min,
       max_epochs = params.max,
     }
     local pocket_train = function()
       local tr_loss = train_dataset(trainer, train_data, bsize, NROWS)
       local va_loss = validate_dataset(trainer, val_data, bsize, NROWS)
       return trainer,tr_loss,va_loss
     end
     --
     local clock = util.stopwatch()
     while pocket:execute(pocket_train) do
       print(pocket:get_state_string(),
             "|w|= "..common.print_weights_norm2(trainer, "w.*"),
             "|b|= "..common.print_weights_norm2(trainer, "b.*"),
             "nump= " .. trainer:get_optimizer():get_count())
       io.stdout:flush()
     end
     local state = pocket:get_state_table()
     local best_val = state.best_val_error
     local best_model = state.best:clone()
     print("# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
     print("# BEST VAL ERROR = ", best_val)
     local best = best_model
     table.insert(models, best)
     return best
   end,
   -- classify function
   classify
)
---------------------------------------------------------------------------
print("# NMODELS TRAINED = ", #models)
print("# LOSS:",table.unpack(MV))
print("# AUC:",table.unpack(AUC))
assert(#models > 0, "None model has been trained :S")
io.stdout:flush()
---------------------------------------------------------------------------
-- save models to output prefix
util.serialize({ params=params, models=models }, "%s/%s.net"%{PREFIX,SUBJECT})

-- perform test prediction if given test output filename
if params.test then
  print("# Loading test")
  local test_data,names = common.load_data(params.fft, "*test*", params)
  test_data.input_dataset = common.apply_std(test_data.input_dataset,means,devs)
  print("# Test num patterns = ", test_data.input_dataset:numPatterns())
  assert(all_train_data.input_dataset:patternSize() == test_data.input_dataset:patternSize(),
         "Different input size between training and test data")
  common.save_test(params.test, names, test_data.input_dataset,
                   params,
                   function(pat, ...)
                     assert(pat:dim(1) % NROWS == 0)
                     local models = ...
                     local result = matrix(pat:dim(1)/NROWS,1):zeros()
                     for _,m in ipairs(models) do
                       local out = classify(pat, m)
                       if log_scale then out:exp() end
                       result:axpy(1.0, out)
                     end
                     result:scal(1/#models)
                     if log_scale then result:log() end
                     return result
                   end,
                   models)
  return AUC[1],test_data.input_dataset:numPatterns()
else
  return AUC[1],all_train_data.input_dataset:numPatterns()
end
