local f = {}
local c = {}
for i=1,#arg,2 do
  c[#c+1] = assert(tonumber(arg[i]), "Needs a number at " .. tostring(i))
  assert(arg[i+1])
  f[#f+1] = assert( io.open(arg[i+1]), "Unable to open " .. arg[i+1] )
  f[#f]:read("*l")
end
local K = iterator(table.ivalues(c)):reduce(math.add(),0)
print("clip,preictal")
local finished = false
while true do
  local prob,id = 0
  local lines = iterator(ipairs(f)):select(2):call("read","*l"):enumerate()
  for i,line in lines do
    if not line then
      finished=true
    else
      id,p = line:match("(.+)%,(.+)")
      prob = prob + c[i]*tonumber(p)
    end
  end
  if not id then break end
  printf("%s,%g\n", id, prob/K)
end
