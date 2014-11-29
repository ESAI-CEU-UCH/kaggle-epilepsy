april_print_script_header(arg)
--
package.path = package.path .. ";/home/PRIVATE/experimentos/KAGGLE/EPILEPSY_PREDICTION/PAKO/scripts/?.lua"
-- library loading and name import
local common = require "common2"
local cmd    = require "cmd"
--
local cmd_opt_parser = cmdOpt{
  program_name = arg[0]:basename(),
  argument_description = "",
  main_description = "Training program for Kaggle epilepsy challenge",
}
--
cmd.add_defopt(cmd_opt_parser)
cmd.add_cv(cmd_opt_parser)
cmd.add_mlp(cmd_opt_parser)
cmd.add_sgd(cmd_opt_parser)
cmd.add_EM(cmd_opt_parser)
cmd.add_supervised(cmd_opt_parser)
cmd.add_seeds(cmd_opt_parser)
cmd.add_noise(cmd_opt_parser)
cmd.add_help(cmd_opt_parser)
--
local params = cmd_opt_parser:parse_args(arg,"defopt")
---------------------------------------------------------------------------
local test_list = params.list:gsub("train$","test")
local weights_random = random(params.wseed)
local shuffle_random = random(params.sseed)
local perturbation_random = random(params.pseed)
local perturbation_random_2 = random(params.pseed2)
local perturbation_random_3 = random(params.pseed3)
local log_scale,smooth,loss_param = common.loss_stuff(params.loss)
local loss_function = ann.loss[params.loss](loss_param)
local SUBJECT = assert( params.list:basename():match("^([^_]+_.).+$"), 
                        "Impossible to determine the subject" )
local PREFIX = params.prefix or ""
-- extension of command line options
if params.bsize == 0 then params.bsize = nil end
params.log_scale = log_scale
params.PREFIX = PREFIX
params.SUBJECT = SUBJECT
--
print("# Loading training")
local all_train_data,list_names = common.load_data(params.list, params)
local means,devs = common.compute_means_devs(all_train_data.input_dataset)
all_train_data.input_dataset = common.apply_std(all_train_data.input_dataset,
                                                means, devs)
local input_size = all_train_data.input_dataset:patternSize()
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
--
local the_net,weights,isize

if params.pretrain and params.h1>0 then
  print("# PRETRAIN")
  local layers_table = { { size=input_size, actf="linear" } }
  table.insert(layers_table, { size=params.h1, actf=params.actf })
  if params.h2 > 0 then
    table.insert(layers_table, { size=params.h2, actf=params.actf })
  end
  if params.h3 > 0 then
    table.insert(layers_table, { size=params.h3, actf=params.actf })
  end
  if params.h4 > 0 then
    table.insert(layers_table, { size=params.h4, actf=params.actf })
  end
  if params.h5 > 0 then
    table.insert(layers_table, { size=params.h5, actf=params.actf })
  end
  if params.h6 > 0 then
    table.insert(layers_table, { size=params.h6, actf=params.actf })
  end
  params_pretrain = {
    input_dataset         = dataset.token.filter(all_input_ds, ann.components.flatten()),
    replacement           = 256,
    shuffle_random        = random(1234),
    weights_random        = random(7890),
    layers                = layers_table,
    bunch_size            = 256,
    optimizer             = function() return ann.optimizer.sgd() end,
    -- training parameters
    training_options      = {
      global = {
        ann_options = { learning_rage = 0.1, momentum = 0.2, weight_decay = 0.001 },
        noise_pipeline = {
          function(ds) return dataset.token.filter(ds, ann.components.gaussian_noise{ random=perturbation_random_2, var=0.1 }) end,
          function(ds) return dataset.token.filter(ds, ann.components.salt_and_pepper{ one=0, zero=0, prob=0.2, random=perturbation_random_2 }) end,
        },
        min_epochs = 10 * math.ceil(all_input_ds:numPatterns() / 256),
        max_epochs = 20 * math.ceil(all_input_ds:numPatterns() / 256),
        pretraining_percentage_stopping_criterion = 0.1,
      },
    }
  }
  sdae_table,deep_classifier = ann.autoencoders.greedy_layerwise_pretraining(params_pretrain)
  the_net = deep_classifier
  weights = the_net:copy_weights()
  -- dropout
  if params.dropout > 0.0 then
    the_net = ann.components.stack()
    for i=1,deep_classifier:size() do
      local component = deep_classifier:get(i)
      the_net:push(component)
      if component:get_name():match("^actf") then
        print("# DROPOUT ", component:get_name())
        the_net:push( ann.components.dropout{ random=perturbation_random,
                                              prob=params.dropout } )
      end
    end
    weights:scal( 1/(1-params.dropout) )
  end
else
  print("# NO PRETRAIN")
  the_net,isize = common.build_mlp_extractor_EM{ input = input_size,
                                                 actf = params.actf,
                                                 layers = { params.h1, params.h2, params.h3, params.h4, params.h5, params.h6 },
                                                 dropout = params.dropout,
                                                 perturbation_random = perturbation_random,
                                                 zca = params.zca,
                                                 pca = params.pca,
                                                 train_ds = all_input_ds }
end
all_input_ds = nil
collectgarbage("collect")

common.push_logistic_layer(the_net, { input = isize,
                                      log_scale = log_scale, })

---------------------------
-- CROSS-VALIDATION LOOP --
---------------------------
local models = {}
local CV = common.train_with_crossvalidation
local AUC,MV = CV(list_names, all_train_data, params,
   -- train function
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
     -- threshold for training expectation step
     -- threshold for training expectation step
     local best_model,best_val,TH = nil,math.huge,nil
     for _,TH in ipairs{ 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5 } do
       collectgarbage("collect")
       --
       local trainer = trainable.supervised_trainer(the_net:clone(),
                                                    loss_function,
                                                    bsize,
                                                    ann.optimizer.sgd(),
                                                    smooth)
       local _,w = trainer:build{ input = input_size, output = 1,
                                  weights = (weights and weights:clone()) or nil }
       print("# MODEL SIZE = ", trainer:size())
       if not params.pretrain or params.h1==0 then
         print("# RANDOMIZE WEIGHTS")
         trainer:randomize_weights{ inf    = -params.wrange,
                                    sup    =  params.wrange,
                                    random =  weights_random,
                                    use_fanin  = true,
                                    use_fanout = true }
         if params.actf == "relu" then
           for _,b in trainer:iterate_weights("b.*") do b:fill(0.1) end
         end
       else
         print("# RANDOMIZE OUTPUT")
         trainer:randomize_weights{ inf    = -params.wrange,
                                    sup    =  params.wrange,
                                    random =  weights_random,
                                    use_fanin  = true,
                                    use_fanout = true,
                                    name_match = ".N" }
         print("#", trainer:norm2("w[123456789]"), trainer:norm2("b[123456789]"))
         assert(trainer:norm2() < 10)
         if params.actf == "relu" then
           trainer:weights("bN"):fill(0.1)
         end
       end
       
       if params.mnp then
         trainer:set_option("max_norm_penalty", params.mnp)
         trainer:set_layerwise_option("b.*", "max_norm_penalty", 0.0)
       end
       if params.wd then
         if params.reg_last_layer then
           trainer:set_layerwise_option("wN", "weight_decay", params.wd)
         else
           trainer:set_option("weight_decay", params.wd)
           trainer:set_layerwise_option("b.*", "weight_decay", 0.0)
         end
       end
       if params.l1 then
         if params.reg_last_layer then
           trainer:set_layerwise_option("wN", "L1_norm", params.l1)
         else
           trainer:set_option("L1_norm", params.l1)
           trainer:set_layerwise_option("b.*", "L1_norm", 0.0)
         end
       end
       if params.decay then
         trainer:set_option("decay", params.decay)
       end
       trainer:set_option("learning_rate", params.lr)
       trainer:set_option("momentum", params.mt)
       
       local pocket = trainable.train_holdout_validation{
         stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(1.5),
         min_epochs = params.min,
         max_epochs = params.max,
       }
       local train = function()
         local tr_loss = trainer:train_dataset(train_data)
         local va_loss
         if params.combine then
           va_loss = common.validate_EM(classify, val_data,
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
       print("# MAXIMIZATION STEP")
       while pocket:execute(train) do
         print(pocket:get_state_string(),
               "|w|= "..common.print_weights_norm2(trainer, "w.*"),
               "|b|= "..common.print_weights_norm2(trainer, "b.*"),
               "nump= " .. trainer:get_optimizer():get_count())
         -- if pocket:is_best() then trainer:save("wop.lua", "ascii") end
         io.stdout:flush()
       end
       local state = pocket:get_state_table()
       if state.best_val_error < best_val then
         best_val = state.best_val_error
         best_model = state.best:clone()
         best_TH = TH
         break
       else
         break
       end
       print("# EXPECTATION STEP")
       print("# TH = ", TH)
       train_data.output_dataset = common.expectation(classify,
                                                      train_data.input_dataset,
                                                      train_data.output_dataset,
                                                      TH, log_scale,
                                                      best_model)
       if not train_data.output_dataset then break end
       print("# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
     end
     print("# BEST VAL ERROR = ", best_val, best_TH)
     local best = best_model
     table.insert(models, best)
     return best
   end,
   -- classify function
   classify
)
---------------------------------------------------------------------------
print("# USING NMODELS = ", #models)
print("# LOSS:",table.unpack(MV))
print("# AUC:",table.unpack(AUC))
assert(#models > 0, "None model has been selected")
io.stdout:flush()
---------------------------------------------------------------------------
util.serialize(models, "%s%s.net"%{PREFIX,SUBJECT})

if params.test then
  print("# Loading test")
  if params.cor then params.cor = params.cor:gsub(".train", ".test") end
  local test_data,names = common.load_data(test_list,params)
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
