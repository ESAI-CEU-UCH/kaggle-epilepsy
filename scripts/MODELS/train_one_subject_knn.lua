april_print_script_header(arg)
--
package.path = package.path .. ";/home/PRIVATE/experimentos/KAGGLE/EPILEPSY_PREDICTION/PAKO/scripts/?.lua"
-- library loading and name import
local common = require "common2"
local cmd    = require "cmd"
local posteriorKNN = knn.kdtree.posteriorKNN
---------------------------------------------------------------------------
-- command line options
local cmd_opt_parser = cmdOpt{
  program_name = arg[0]:basename(),
  argument_description = "",
  main_description = "Training program for Kaggle epilepsy challenge",
}
cmd.add_defopt(cmd_opt_parser)
cmd.add_cv(cmd_opt_parser)
cmd.add_knn(cmd_opt_parser)
cmd.add_help(cmd_opt_parser)
local params = cmd_opt_parser:parse_args(arg,"defopt")
---------------------------------------------------------------------------
-- extension of command line options
if params.bsize == 0 then params.bsize = nil end
local log_scale = true
params.log_scale = log_scale
--
local SUBJECT = assert( params.list:basename():match("^([^_]+_.).+$"), 
                        "Impossible to determine the subject" )
local PREFIX = params.prefix or ""
--
params.PREFIX = PREFIX
params.SUBJECT = SUBJECT
---------------------------------------------------------------------------
local test_list = params.list:gsub("train$","test")
local rnd = random(params.seed)

print("# Loading training")
local all_train_data,list_names = common.load_data(params.list, params)

local means,devs = common.compute_means_devs(all_train_data.input_dataset)
-- matrix(#means,1,means):toTabFilename("means2")
-- matrix(#devs,1,devs):toTabFilename("devs2")
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
  -- mat:toTabFilename("jarl2.mat")
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
local AUC,MV = CV(list_names, all_train_data, params,
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

if params.test then
  local kdt = train_knn(rnd, all_train_data.input_dataset)
  print("# Loading test")
  if params.cor then params.cor = params.cor:gsub(".train", ".test") end
  local test_data,names = common.load_data(test_list,params)
  test_data.input_dataset = common.apply_std(test_data.input_dataset,means,devs)
  print("# Test num patterns = ", test_data.input_dataset:numPatterns())
  local tr_out_ds = all_train_data.output_dataset
  common.save_test(params.test, names, test_data.input_dataset, params,
                   classify_matrix, kdt, tr_out_ds)
  return AUC[1],test_data.input_dataset:numPatterns()
else
  return AUC[1],all_train_data.input_dataset:numPatterns()
end
