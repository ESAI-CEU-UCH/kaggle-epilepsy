local rnd = random(12384)
local log_scale = false
local norm = false
local NaN = mathcore.limits.float.quiet_NaN()
local data = {}
local SAMPLES = 2000 --500
local SUBJECTS = { "Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5",
                   "Patient_1", "Patient_2" }
local result_test = { {'clip','preictal'} }

local mode = "linear" -- table.remove(arg,1)
assert(mode == "logistic" or mode == "linear" or mode == "mlp" or mode == "geom",
       "Needs 'geom', 'logistic', 'linear' or 'mlp' mode as first argument")

local function normalize(w)
  for _,row in matrix.ext.iterate(w,1) do row:abs():scal( 1/row:sum() ) end
end

local averaged_AUC = 0
local total = 0
-- for each subject do
for _,subject in ipairs(SUBJECTS) do
  -- print("#",subject)
  collectgarbage("collect")
  local models = {}
  -- for each model output
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
  -- AUTODIFF
  local shared = { w1 = matrix(1, #models) }
  local log, T, M = autodiff.op.log, autodiff.op.transpose, autodiff.matrix
  local i,w1 = M("i w1")
  local f = log( i * T( w1 ) )
  local net = autodiff.ann.model(f, i, { w1 }, shared, #models, 1)
  -- LOSS
  local loss = ann.loss.cross_entropy()
  log_scale = true
  norm = false
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
    -- log-likelyhood
    local loss = trainer:validate_dataset{ input_dataset = in_ds,
                                           output_dataset = out_ds }
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
  -- print(pocket:get_state_string())
  --
  local val_input = in_ds:toMatrix()
  local val_output = trainer:calculate(val_input)
  local val_target = out_ds:toMatrix()
  -- print("LOSS=", loss:compute_loss(val_output,val_target),
  --       iterator.range(val_input:dim(2)):
  --         map(function(i)
  --             local val = val_input(':',i):clone()
  --             if log_scale then val:log() end
  --             return ( loss:compute_loss(val,val_target) )
  --         end):concat(" "))
  --
  if log_scale then val_output:exp() end
  local auc = metrics.roc(val_output, val_target):compute_area()
  print("# %20s"%{subject}, "AUC=",auc,
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
  if log_scale then test_output:exp() end
  -- print(matrix.join(2, test_output, test_m))
  -- print(test_m)
  -- print(test_output)
  for i=1,#models[1].test do
    table.insert(result_test, { models[1].test[i][1], test_output:get(i,1) })
  end
  --
  averaged_AUC = averaged_AUC + auc*test_output:size()
  total = total + test_output:size()
  print("#\t\t\t\t\t\t", table.concat(trainer:weights("w1"):toTable(), " "))
  -- print( matrix.op.exp( trainer:weights("w1") ) / matrix.op.exp( trainer:weights("w1") ):sum() )
  
  -- if true and ( mode == "geom" or mode == "linear" ) and subject == "Dog_2" then
  --   local w1 = trainer:weights("w1")
  --   for a=-10.0, 10.0, 0.4 do
  --     w1:set(1,1,a)
  --     for b=-10.0, 10.0, 0.4 do
  --       --local b = 1.0 - a
  --       w1:set(1,2,b)
  --       local loss = trainer:validate_dataset{ input_dataset = in_ds,
  --                                              output_dataset = out_ds } + trainer:get_option("weight_decay") * w1:dot(w1) * 0.5
  --       print("LOSS", a, b, loss)
  --     end
  --   end
  --   os.exit()
  -- end

end

print("# AVERAGED AUC =", averaged_AUC/total)

fprintf(io.stderr, iterator(result_test):map(table.unpack):concat(",","\n") )
fprintf(io.stderr, "\n")
