april_print_script_header(arg)
--
local LISTS_DIR = "/home/experimentos/KAGGLE/EPILEPSY_PREDICTION/lists"
--
local script = table.remove(arg,1)
local list_base = table.remove(arg,1)
assert(not script:match("^%-%-"), "Needs the training script as first argument")
assert(not list_base:match("^%-%-"),
       "Needs a base list (without subject, with .train) as second argument")

-- initialize test filename
for i=1,#arg do
  local test = arg[i]:match("--test=(.+)")
  if test then
    print("# INITIALIZING TEST OUTPUT:", test)
    f = io.open(test,"w")
    f:write("clip,preictal\n")
    f:close()
  end
end
--

print("#",script,list_base)
local auc_and_sizes = {}
local train_script = assert(loadfile(script))
local output_filename = os.tmpname()
table.insert(arg, "--append=%s"%{output_filename})
for _,p in ipairs{
  {"Dog_1",16},{"Dog_2",16}, {"Dog_3",16}, {"Dog_4",16}, {"Dog_5",15},
  {"Patient_1",15}, {"Patient_2",24},
} do
  local subject,channels=table.unpack(p)
  local train_list = "%s/%s_%s"%{LISTS_DIR,subject,list_base}
  print("# TRAINING SUBJECT = ", subject)
  print("# TRAIN LIST = ", list_base)
  table.insert(arg, "--list=%s"%{train_list})
  table.insert(arg, "--channels=%d"%{channels})
  print("# CMD = ", table.concat(arg, " "))
  print("####################################################################")
  auc_and_sizes[#auc_and_sizes+1] = table.pack( train_script(arg) )
  print("####################################################################")
  table.remove(arg,#arg)
  table.remove(arg,#arg)
  collectgarbage("collect")
end
local output = matrix.fromTabFilename(output_filename)
os.remove(output_filename)
local result

local major,minor,commit = util.version()

if tonumber(commit) >= 2192 then
  result = stats.boot{
    size = output:dim(1),
    R = 1000,
    statistic = function(indices)
      local sample = output:index(1, indices)
      return metrics.roc(sample(':',1), sample(':',2)):compute_area()
    end,
    ncores = util.omp_get_num_threads(),
  }
else
  result = stats.boot{
    data = output,
    R = 1000,
    statistic = function(it)
      local auc = metrics.roc()
      for _,row in it do auc:add(row(1), row(2)) end
      return auc:compute_area()
    end,
    ncores = util.omp_get_num_threads(),
  }
end

local auc = metrics.roc(output:select(2,1), output:select(2,2)):compute_area()
local a,b = stats.boot.ci(result, 0.95)
local c = stats.boot.percentil(result, 0.5)
local width = (c-a)/2
print("# AUC:",auc, "+- " .. width, a, b, c)
--
local total_size = iterator(table.ivalues(auc_and_sizes)):field(2):reduce(math.add(),0)
local averaged_auc = iterator(table.ivalues(auc_and_sizes)):
map(function(t) return t[1] * t[2]/total_size end):reduce(math.add(),0)
iterator(ipairs(auc_and_sizes)):
  apply(function(k,v)
      print("# Subject " .. k .. ":", v[1], v[2])
  end)
print("# averaged AUC:",averaged_auc)
