package.path = package.path .. ";./scripts/?.lua"
--
local common = require "common"
local data_path = common.DATA_PATH
local output = common.SEQUENCES_PATH
local f = io.open(output, "w")
for _,dir in ipairs{ "Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5",
                     "Patient_1", "Patient_2" } do
  for _,mask in ipairs{"*preictal*", "*interictal*"} do
    local expected_sequence = 0  
    for _,filename in ipairs(glob("%s/%s/%smat"%{data_path,dir,mask})) do
      collectgarbage("collect")
      local m = matlab.read(filename)
      local _,data = next(m)
      local seq = data.sequence:get(1,1)
      if seq ~= expected_sequence+1 then
        printf("# WARNING: Incorrect sequence in filename %s, expected %d, found %d\n",
               filename, expected_sequence+1, seq)
        expected_sequence = seq
      end
      expected_sequence = (expected_sequence + 1) % 6
      fprintf(f, "%s %d\n", filename:basename():gsub(".mat$",""), seq)
      f:flush()
    end
  end
end
f:close()
