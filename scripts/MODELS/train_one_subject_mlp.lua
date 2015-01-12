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
cmd.add_mlp(cmd_opt_parser)
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
local function classify(pat,...)
  local trainer = ...
  return trainer:use_dataset{ input_dataset = dataset.matrix(pat) }:toMatrix()
end
-- build the ANN
local the_net,isize = common.build_mlp_extractor{ input = input_size,
                                                  actf = params.actf,
                                                  layers = { params.h1, params.h2, params.h3, params.h4, params.h5, params.h6 },
                                                  dropout = params.dropout,
                                                  perturbation_random = perturbation_random,
                                                  zca = params.zca,
                                                  pca = params.pca,
                                                  train_ds = all_input_ds }
common.push_logistic_layer(the_net, { input = isize,
                                      log_scale = log_scale, })
collectgarbage("collect")

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
                                                  ann.optimizer.sgd(),
                                                  smooth)
     local _,w = trainer:build{ input = input_size, output = 1 }
     print("# MODEL SIZE = ", trainer:size())
     print("# RANDOMIZE WEIGHTS")
     trainer:randomize_weights{ inf    = -params.wrange,
                                sup    =  params.wrange,
                                random =  weights_random,
                                use_fanin  = true,
                                use_fanout = true }
     if params.actf == "relu" then
       for _,b in trainer:iterate_weights("b.*") do b:fill(0.1) end
     end
     -- set learning hyper-parameters
     if params.mnp then
       trainer:set_option("max_norm_penalty", params.mnp)
       trainer:set_layerwise_option("b.*", "max_norm_penalty", 0.0)
     end
     if params.wd then
       trainer:set_option("weight_decay", params.wd)
       trainer:set_layerwise_option("b.*", "weight_decay", 0.0)
     end
     if params.l1 then
       trainer:set_option("L1_norm", params.l1)
       trainer:set_layerwise_option("b.*", "L1_norm", 0.0)
     end
     if params.decay then
       trainer:set_option("decay", params.decay)
     end
     trainer:set_option("learning_rate", params.lr)
     trainer:set_option("momentum", params.mt)
     -- pocket algorithm object
     local pocket = trainable.train_holdout_validation{
       stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(1.5),
       min_epochs = params.min,
       max_epochs = params.max,
     }
     local pocket_train = function()
       local tr_loss = trainer:train_dataset(train_data)
       local va_loss
       if params.combine then
         va_loss = common.validate(classify, val_data,
                                   trainer:get_loss_function(),
                                   NROWS, log_scale, params.combine,
                                   trainer)
       else
         va_loss = trainer:validate_dataset(val_data)
       end
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
  local test_data,names = common.load_data(params.fft, common.TEST_MASK, params)
  test_data.input_dataset = common.apply_std(test_data.input_dataset,means,devs)
  print("# Test num patterns = ", test_data.input_dataset:numPatterns())
  assert(all_train_data.input_dataset:patternSize() == test_data.input_dataset:patternSize(),
         "Different input size between training and test data")
  common.save_test(params.test, names, test_data.input_dataset,
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
  return AUC[1],test_data.input_dataset:numPatterns()
else
  return AUC[1],all_train_data.input_dataset:numPatterns()
end
