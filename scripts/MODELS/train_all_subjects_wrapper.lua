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
print("#",script,list_base,cor_base)
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
local total_size = iterator(table.ivalues(auc_and_sizes)):field(2):reduce(math.add(),0)
local averaged_auc = iterator(table.ivalues(auc_and_sizes)):
map(function(t) return t[1] * t[2]/total_size end):reduce(math.add(),0)
print("# averaged AUC:",averaged_auc)
