local common = {}

local conf = iterator(io.lines("scripts/conf.txt")):
map(function(line)
      local k,v=line:match("^export ([^%s]+)%s*=%s*([^%s]+)$")
      return k,v
end):table()
for k,v in pairs(conf) do common[k] = v end

local DATA_PATH       = conf.DATA_PATH
local SEQUENCES_PATH  = conf.SEQUENCES_PATH

local mop   = matrix.op
local amean = stats.amean
local gmean = stats.gmean
local hmean = stats.hmean

local major,minor,commit = util.version()

if tonumber(commit) >= 2192 then
  -- avoid warning messages of deprecated functions
  matrix.col_major = matrix
  matrix.row_major = matrix
  class.extend(matrix, "get_major_order", function() return "row_major" end)
  stats.mean_var = stats.running.mean_var
end

------------------------------------------------------------------------------
---------------------- CROSS VALIDATION PARTITIONS ---------------------------
------------------------------------------------------------------------------

-- load all the sequence numbers taken from original mat files and returns a
-- data_0,data_1 tables of pairs {filename,sequence_number}
local function load_sequences(filtered_names)
  local f = assert( io.open("/home/experimentos/CORPORA/KAGGLE/EPILEPSY_PREDICTION/SEQUENCES.txt") )
  local data_0 = {}
  local data_1 = {}
  for line in f:lines() do
    local filename,seq = line:match("^([^%s]+)%s+([^%s]+)$")
    if not filtered_names or filtered_names[filename] then
      local seq = tonumber(seq)
      table.insert( filename:match("interictal") and data_0 or data_1,
                    { filename, seq } )
    end
  end
  return data_0,data_1
end

-- compute the cross validation partition for a given subject name (Dog_1,
-- Dog_2, ...) and the pair data_0,data_1 returned by load_sequences function.
-- It returns blocks, a table of filename arrays.
local function compute_cross_validation_partition(subject, data_0, data_1)
  assert(type(subject) == "string")
  assert(type(data_0) == "table" and type(data_1) == "table")
  -- Receives the list with all loaded data filenames and sequences, and returns
  -- a table of sequences.
  local function divide_sequences(data) -- up-values: subject
    local prev_seq  = math.huge
    local sequences = {}
    for _,pair in ipairs(data) do
      local filename,seq = table.unpack(pair)
      if filename:match(subject) then
        if prev_seq > seq then
          table.insert(sequences, { filename })
        else
          table.insert(sequences[#sequences], filename)
        end
        prev_seq = seq
      end
    end
    assert(#sequences > 0)
    table.sort(sequences, function(a,b) return #a > #b end)
    return sequences
  end
  
  -- Computes the distribution of sequences into cross-validation blocks.
  -- Receives a table of blocks (empty or not), a table of sequences,
  -- and the desired number of blocks.
  local function distribute_into_blocks(blocks, sequences, num_blocks)
    assert(type(blocks) == "table" and type(sequences) == "table")
    assert(type(num_blocks) == "number")
    local seq = 0
    while seq < #sequences do
      local k = seq % num_blocks + 1
      seq = seq + 1
      for j=1,#sequences[seq] do
        blocks[k] = table.insert(blocks[k] or {}, sequences[seq][j])
      end
    end
  end

  local sequences_0 = divide_sequences(data_0)
  local sequences_1 = divide_sequences(data_1)
  
  local num_blocks = iterator(table.ivalues(sequences_1)):
  map(function(v) return #v end):
  filter(function(n) return n == 6 end):enumerate():select(1):reduce(math.max,0)
  -- if num_blocks > 3 then num_blocks = math.floor(num_blocks / 2) end
  -- num_blocks = math.min(num_blocks, 6)
  
  -- distribute the partition
  local blocks = {}
  distribute_into_blocks(blocks, sequences_0, num_blocks)
  distribute_into_blocks(blocks, sequences_1, num_blocks)
  assert(#blocks == num_blocks)
  
  -- sanity check: non-repeated filenames
  local dict,N = {},0
  for k,b in ipairs(blocks) do
    for j,filename in ipairs(b) do
      dict[filename] = not assert(not dict[filename])
      N=N+1
    end
  end
  local function count(s)
    return iterator(table.ivalues(s)):map(function(v) return #v end):reduce(math.add(),0)
  end
  assert(N == count(sequences_0) + count(sequences_1))
  --
  return blocks
end

-- Given a list of train filenames, a table all_train_data with
-- input_dataset,output_dataset, and a train_func, a sequential iteration over
-- all CV partitions will be performed, calling train_func with the
-- tr_data,va_data for each of the partitions.
function common.train_with_crossvalidation(list_names, all_train_data, params,
                                           train_func, classify_func)
  local NROWS = all_train_data.input_dataset:numPatterns()/#list_names
  print("# CV INPUT NUM ROWS = ", NROWS)
  assert(math.floor(NROWS) == NROWS)
  local name2id = table.invert(list_names)
  local subject = list_names[1]:match("^([a-zA-Z]+_[0-9]+).*$")
  local data_0,data_1 = load_sequences(name2id)
  local blocks = compute_cross_validation_partition(subject, data_0, data_1)
  -- separate data_0,data_1 to create multiple CV partitions
  local NUM_BLOCKS = #blocks
  local blocks_idxs = {}
  for k,b in ipairs(blocks) do
    blocks_idxs[k] = {}
    for j,name in ipairs(b) do
      blocks_idxs[k][j] = april_assert( name2id[name],
                                        "Name not found: %s", name )
    end
  end
  print("# NBLOCKS = ", NUM_BLOCKS)
  --
  -- up-value: blocks_idxs, all_train_data
  local function separate_blocks(K,psz)
    -- up-value: blocks_idxs, all_train_data
    local function separate(ini,last,data)
      local data = data or { input_dataset={}, output_dataset={} }
      --
      local function insert_slice(where, first_index, last_index)
        table.insert(assert(data[where]),
                     dataset.slice(assert(all_train_data[where]),
                                   first_index, last_index))
      end
      --
      for k=ini,last do
        for j,idx in ipairs(blocks_idxs[k]) do
          local first_index = (idx-1)*NROWS + 1
          local last_index = idx*NROWS
          assert( ((first_index-1) % NROWS) == 0)
          assert( (last_index % NROWS) == 0)
          insert_slice("input_dataset", first_index, last_index)
          insert_slice("output_dataset", first_index, last_index)
        end
      end
      return data
    end
    -- training data contains all blocks except K
    local tr_data = separate(1,   K-1,          tr_data)
    local tr_data = separate(K+1, #blocks_idxs, tr_data)
    -- validation data contains only block K
    local va_data = separate(K,   K,            va_data)
    return tr_data,va_data
  end
  --
  local mv = stats.mean_var()
  local roc_curve = metrics.roc()
  local P = 1
  --
  local val_results = {}
  local psz = all_train_data.input_dataset:patternSize()
  local nump = all_train_data.input_dataset:numPatterns()
  -- for every possible CV partition
  for k=1,NUM_BLOCKS do
    -- take block k for validation, the rest for training
    local tr_data,va_data = separate_blocks(k, psz)
    tr_data.input_dataset = dataset.union(tr_data.input_dataset)
    tr_data.output_dataset = dataset.union(tr_data.output_dataset)
    va_data.input_dataset = dataset.union(va_data.input_dataset)
    va_data.output_dataset = dataset.union(va_data.output_dataset)
    -- verbose output
    print("# -------------------------------------------------")
    print("# BLOCK ", k, "of", NUM_BLOCKS)
    local tr_nump = tr_data.input_dataset:numPatterns()
    local va_nump = va_data.input_dataset:numPatterns()
    print("# TR nump = ", tr_nump)
    print("# VA nump = ", va_nump)
    assert(tr_nump+va_nump == nump)
    -- TRAIN
    local args = { train_func(tr_data, va_data) }
    -- VALIDATE
    local input = va_data.input_dataset:toMatrix()
    local out,out2 = classify_func(input, table.unpack(args) )
    local tgt,tgt2 = va_data.output_dataset:toMatrix()
    if params.log_scale then out = out:clone():exp() end
    -- remove zero values
    -- out:cmul(out:clone():gt(1e-20))
    if params.combine then
      out2 = common.combine_filename_outputs(out, NROWS, params.combine)
      tgt2 = common.combine_filename_outputs(tgt, NROWS, "max", true)
    else
      -- sanity check, ignore its result
      common.combine_filename_outputs(tgt, NROWS, "max", true)
      out2,tgt2 = out,tgt
    end
    out2:clamp(0,1)
    tgt2:clamp(0,1)
    -- current partition partial result
    local partial = metrics.roc(out2,tgt2):compute_area()
    print("# partial",partial)
    mv:add(partial)
    -- accumulate for subject ROC curve
    roc_curve:add(out2,tgt2)
    table.insert(val_results, {out2,tgt2})
    -- append validation result
    local a0,a1,v0,v1 = common.append(params.append, out2, tgt2)
    -- verbose output
    if params.save_activations then
      assert( os.execute("mkdir -p %s"%{params.save_activations}) )
      local preictal = {}
      local interictal = {}
      for i=1,out:dim(1) do
        table.insert(tgt:get(i,1)==1 and preictal or interictal, out:get(i,1))
      end
      assert(#preictal % NROWS == 0)
      assert(#interictal % NROWS == 0)
      local preictal = matrix(preictal)
      local interictal = matrix(interictal)
      --
      preictal:rewrap(preictal:size()/NROWS, NROWS):transpose():
        toTabFilename("%s/PREICTAL.%02d.txt"%{params.save_activations,P})
      --
      interictal:rewrap(interictal:size()/NROWS, NROWS):transpose():
        toTabFilename("%s/INTERICTAL.%02d.txt"%{params.save_activations,P})
    end
    io.stdout:flush()
    P=P+1
  end
  -- roc_curve:compute_curve():toTabFilename("curve.txt")
  local outm = matrix.join(1,iterator(val_results):field(1):table())
  local tgtm = matrix.join(1,iterator(val_results):field(2):table())
  local val_result = matrix.join(2,outm,tgtm)
  val_result:toTabFilename("%svalidation_%s.txt"%{params.PREFIX,params.SUBJECT})
  return {roc_curve:compute_area()},{mv:compute()}
end

------------------------------------------------------------------------------
--------------------------- PREPROCESSING ------------------------------------
------------------------------------------------------------------------------

function common.load_matlab_file(filename)
  local ok,m,hz,num_channels = xpcall(function()
      local data = assert( matlab.read(filename) )
      local name,tbl = assert( next(data) )
      local hz = assert( tbl.sampling_frequency:get(1,1) )
      local m = assert( tbl.data:to_float() )
      return m,hz,m:dim(1)
                         end,
    debug.traceback)
  if not ok then
    print(m)
    error("An error happened processing %s"%{filename})
  end
  return m,hz,num_channels
end

function common.compute_fft(m, hz, wsize, wadvance)
  local wsize,wadvance = math.floor(wsize*hz),math.floor(wadvance*hz)
  local fft_tbl = {}
  -- assert( m:dim(2)/wadvance - math.floor(m:dim(2)/wadvance) < 0.1 )
  for i=1,m:dim(1) do
    fft_tbl[i] = matrix.ext.real_fftwh(m:select(1,i), wsize, wadvance)
    if i > 1 then
      assert( fft_tbl[i]:dim(1) == fft_tbl[i-1]:dim(1) )
      assert( fft_tbl[i]:dim(2) == fft_tbl[i-1]:dim(2) )
    end
  end
  assert( #fft_tbl == m:dim(1) )
  return fft_tbl
end

function common.compress(m)
  return m:clone():clamp(1.0, (m:max())):log()
end

function common.make_prep_function(HZ,FFT_SIZE,WSIZE,WADVANCE,out_dir,filter)
  return function(mat_filename)
    local out_filename = "%s.channel_%02d.csv.gz" %
      { (mat_filename:basename():gsub("%.mat", "" )), 1, }
    if not io.open(out_dir .. out_filename) then
      print("#",mat_filename)
      collectgarbage("collect")
      local m,hz = common.load_matlab_file(mat_filename)
      assert( math.abs(hz - HZ) < 1 )
      local fft_tbl = common.compute_fft(m, hz, WSIZE, WADVANCE)
      local bf_tbl = {}
      for i=1,#fft_tbl do
        local out_filename = "%s.channel_%02d.csv.gz" %
          { (mat_filename:basename():gsub("%.mat", "" )), i, }
        assert( fft_tbl[i]:dim(2) == FFT_SIZE, fft_tbl[i]:dim(2) )
        bf_tbl[i] = filter( fft_tbl[i] )
        bf_tbl[i]:toTabFilename(out_dir .. out_filename)
      end
    end
  end
end

function common.compute_PLOS_filter(HZ, FFT_SIZE, NUM_BF)
  assert(NUM_BF == 6)
  local BIN_WIDTH = 0.5*HZ / FFT_SIZE
  -- create a bank filter matrix (sparse matrix)
  local limits = {
    {  0.1,  4 }, -- delta
    {  4,    8 }, -- theta
    {  8,   12 }, -- alpha
    { 12,   30 }, -- beta
    { 30,   70 }, -- low-gamma
    { 70,  180 }, -- high-gamma
  }
  local filter = matrix(FFT_SIZE,NUM_BF):zeros()
  for i=1,NUM_BF do
    local ini = math.ceil(limits[i][1] / BIN_WIDTH)
    local fin = math.floor(limits[i][2] / BIN_WIDTH)
    local sz  = fin - ini + 1
    local bf  = filter:select(2,i)
    bf({ ini , fin }):fill(1/sz)
  end
  local filter = matrix.sparse.csc(filter)
  return function(m)
    local out = common.compress( m * filter )
    assert(out:dim(1) == m:dim(1))
    assert(out:dim(2) == NUM_BF)
    return out
  end
end

function common.compute_PLOS_0_filter(HZ, FFT_SIZE, NUM_BF)
  assert(NUM_BF == 7)
  local BIN_WIDTH = 0.5*HZ / FFT_SIZE
  -- create a bank filter matrix (sparse matrix)
  local limits = {
    {  0.1,  4 }, -- delta
    {  4,    8 }, -- theta
    {  8,   12 }, -- alpha
    { 12,   30 }, -- beta
    { 30,   70 }, -- low-gamma
    { 70,  180 }, -- high-gamma
  }
  local filter = matrix(FFT_SIZE,NUM_BF):zeros()
  filter:select(2,1):set(1,1)
  for i=1,NUM_BF-1 do
    local ini = math.ceil(limits[i][1] / BIN_WIDTH)
    local fin = math.floor(limits[i][2] / BIN_WIDTH)
    local sz  = fin - ini + 1
    local bf  = filter:select(2,i+1)
    bf({ ini , fin }):fill(1/sz)
  end
  local filter = matrix.sparse.csc(filter)
  return function(m)
    local out = common.compress( m * filter )
    assert(out:dim(1) == m:dim(1))
    assert(out:dim(2) == NUM_BF)
    return out
  end
end

function common.compute_rectangular_nonoverlap_filter(HZ, FFT_SIZE, NUM_BF)
  local BF_WIDTH = FFT_SIZE / NUM_BF
  assert( BF_WIDTH == math.floor(BF_WIDTH) )
  -- create a bank filter matrix (sparse matrix)
  local filter = matrix(FFT_SIZE,NUM_BF):zeros()
  for i=1,NUM_BF do
    local bf = filter:select(2,i)
    bf({ BF_WIDTH * (i-1) + 1 , BF_WIDTH * i }):fill(1/BF_WIDTH)
  end
  assert( filter:sum(1) == matrix(1,NUM_BF):ones() )
  local filter = matrix.sparse.csc(filter)
  assert( filter:non_zero_size() == FFT_SIZE )
  return function(m)
    local out = common.compress( m * filter )
    assert(out:dim(1) == m:dim(1))
    assert(out:dim(2) == NUM_BF)
    return out
  end
end

function common.compute_log_triangular_overlap_filter(HZ, F_MIN, F_MAX,
                                                      FFT_SIZE, NUM_BF)
  --local function freq2log(fHz) return 2595.0 * math.log10( 1 + fHz / 700.0 ) end
  --local function log2freq(fHz) return 700.0 * ( 10.0^( fHz / 2595.0) -1.0) end
  local function freq2log(fHz) return 10.0 * math.log10( 1 + fHz / 4.0 ) end
  local function log2freq(fHz) return 4.0 * ( 10.0^( fHz / 10.0 ) - 1.0 ) end
  --
  local log_min = freq2log(F_MIN)
  local log_max = freq2log(F_MAX)
  local BIN_WIDTH = 0.5*HZ / FFT_SIZE
  local LOG_WIDTH = (log_max - log_min) / (NUM_BF + 1)
  -- create a bank filter matrix (sparse matrix)
  local filter = matrix(FFT_SIZE,NUM_BF):zeros()
  local centers = {}
  for i=1,NUM_BF+2 do centers[i] = log2freq(log_min + (i-1)*LOG_WIDTH) end
  for i=1,NUM_BF do
    local bf = filter:select(2,i)
    -- compute triangle
    local start  = math.min(math.round(centers[i] / BIN_WIDTH)+1, FFT_SIZE)
    local center = math.min(math.round(centers[i+1] / BIN_WIDTH)+1, FFT_SIZE)
    local stop   = math.min(math.round(centers[i+2] / BIN_WIDTH)+1, FFT_SIZE)
    -- compute bandwidth (number of indices per window)
    local inc = 1.0/(center - start)
    local dec = 1.0/(stop - center)
    -- compute triangle-shaped filter coefficients
    bf({start,center-1}):linear(0):scal(inc) -- left ramp
    bf({center,stop-1}):linear(0):scal(dec):complement() -- right ramp
    -- normalize the sum of coefficients:
    bf:scal(1/bf:sum())
    -- check for nan or inf
    assert(bf:isfinite(), "Fatal error on log filter bank")
  end
  local filter = matrix.sparse.csc(filter)
  return function(m)
    local out = common.compress( m * filter )
    assert(out:dim(1) == m:dim(1))
    assert(out:dim(2) == NUM_BF)
    return out
  end
end

-----------------------------------------------------------------------------
--------------------------- TRAINING ----------------------------------------
-----------------------------------------------------------------------------

local function get_label_of(basename)
  if basename:match("preictal") then return 1
  elseif basename:match("interictal") then return 0
  else return nil
  end
end

function common.load_data(list,num_channels)
  local input_ds
  local labels = {}
  local list_names = {}
  local ncols,nrows
  for path in io.lines(list) do
    table.insert(list_names, path:basename())
    local mat_tbl = {}
    for j=1,num_channels do
      local filename = "%s.channel_%02d.csv.gz"%{path,j}
      local f = april_assert(io.open(filename), "Unable to open %s", filename)
      local ok,msg = xpcall(function()
          mat_tbl[#mat_tbl + 1] = matrix.read(f,
                                              { [matrix.options.order] = "col_major",
                                                [matrix.options.tab]   = true,
                                                [matrix.options.ncols] = ncols,
                                                [matrix.options.nrows] = nrows, })
      end,
      debug.traceback)
      if not ok then
        print(msg)
        error("Error happened loading %s"%{filename})
      end
      nrows,ncols = table.unpack(mat_tbl[#mat_tbl]:dim())
    end
    local input = matrix.join(2,table.unpack(mat_tbl)):rewrap(nrows,ncols,#mat_tbl)
    local label = get_label_of(path)
    input_ds = input_ds or dataset.token.vector(input:size())
    input_ds:push_back(input)
    if label then labels[#labels + 1] = label end
  end
  if #labels > 0 then assert(#labels == input_ds:numPatterns()) end
  return {
    input_dataset  = input_ds,
    output_dataset = (#labels > 0 and dataset.matrix(matrix(labels))) or nil,
  },
  list_names
end

function common.area_under_curve(trainer, data, log_scale, roc_curve, use_dataset, ...)
  local use_dataset = use_dataset or
    function(trainer,data)
      return trainer:use_dataset(data)
    end
  local tgt = data.output_dataset:toMatrix()
  local out_ds = use_dataset(trainer, { input_dataset = data.input_dataset,
                                        output_dataset = dataset.matrix(matrix(data.input_dataset:numPatterns())) },
                             ...)
  local output = out_ds:toMatrix()
  if log_scale then output:exp() end
  local roc_curve = roc_curve or metrics.roc()
  roc_curve:add(output,tgt)
  --local aux = roc_curve:compute_curve()
  --local cm = stats.confusion_matrix(2)
  --cm:addData(output:clone():gt(0.5):scalar_add(1):toTable(),tgt:clone():scalar_add(1):toTable())
  --aux:toTabFilename("curve.txt")
  return roc_curve:compute_area(),roc_curve,output,tgt
end

function common.compute_means_devs(ds)
  if false then
    local sums = matrix.col_major(ds:patternSize()):zeros():
      rewrap(table.unpack(ds:getPattern(1):dim()))
    local sums2 = sums:clone()
    for i=1,ds:numPatterns() do
      local m = ds:getPattern(i)
      sums:axpy(1.0, m)
      sums2:axpy(1.0, m:clone():pow(2))
    end
    local N = ds:numPatterns()
    local means = sums/N
    local devs = ((sums2 - (sums^2 / N))/(N-1)):sqrt()
    assert((devs:min()) ~= 0)
    return means,devs
  else
    local dim = ds:getPattern(1):dim()
    local N = table.remove(dim,1)
    local sz = ds:patternSize()
    local sums = matrix.col_major(sz/N):zeros():rewrap(table.unpack(dim))
    local sums2 = sums:clone()
    for i=1,ds:numPatterns() do
      local m = ds:getPattern(i) -- :rewrap(N,sz/N)
      for i=1,N do
        local row = m:select(1,i)
        sums:axpy(1.0, row)
        sums2:axpy(1.0, row:clone():pow(2))
      end
    end
    local M = ds:numPatterns()*N
    local means_p = sums/M
    local devs_p = ((sums2 - (sums^2 / M))/(M-1)):sqrt()
    assert((devs_p:min()) ~= 0)
    -- means_p:rewrap(means_p:size(),1):clone("row_major"):toTabFilename("means")
    -- devs_p:rewrap(means_p:size(),1):clone("row_major"):toTabFilename("devs")
    local means = matrix.col_major(N,table.unpack(dim))
    local devs = matrix.col_major(N,table.unpack(dim))
    for i=1,N do
      means:select(1,i):copy(means_p)
      devs:select(1,i):copy(devs_p)
    end
    return means,devs
  end
end

function common.apply_std(ds,means,devs)
  print("# Applying standarization")
  local inv_devs = devs:clone():div(1.0)
  for i=1,ds:numPatterns() do
    local m = ds:getPattern(i)
    m:axpy(-1.0,means)
    m:cmul(inv_devs)
    ds:putPattern(i,m)
  end
end

function common.mean_channel(ds)
  local out_ds
  print("# Computing mean channel")
  for i=1,ds:numPatterns() do
    local m = ds:getPattern(i)
    local s = m:sum(3):rewrap(m:dim(1),m:dim(2))
    local s2 = m:clone():pow(2):sum(3):rewrap(m:dim(1),m:dim(2))
    local out_m = matrix.col_major(m:dim(1), m:dim(2), 1)
    out_m:select(3,1):copy(s):scal(1/m:dim(3))
    --m:select(3,2):copy( ((s2 - (s^2 / m:dim(3))) / (m:dim(3) - 1)) ):sqrt()
    out_ds = out_ds or dataset.token.vector(out_m:size())
    out_ds:push_back(out_m)
  end
  return out_ds
end

local function test_standarization(out, centroids)
  local a0,a1 = table.unpack(centroids)
  assert(a0 and a1)
  local th = 0.5 * (a0 + a1)
  return out:clone():
    map(function(x)
        if x < th then
          return x/th * 0.5
        else -- x >= th
          return 0.5 + (x - th)/(1-th) * 0.5
        end
    end)
end

function common.save_test(output, names, ds, models, log_scale, centroids,
                          forward, ...)
  local forward = forward or
    function(trainer, input)
      local input = input:rewrap(1,table.unpack(input:dim()))
      return trainer:get_component():forward(input)
    end
  assert(not centroids or #centroids==#models)
  local f = assert(io.open(output, "a"))
  for id,pat in ds:patterns() do
    local result
    for j=1,#models do
      local out = forward(models[j], pat, ...):max(1)
      assert(out:size() == 1)
      if log_scale then out = out:clone():exp() end
      if not result then result = matrix.as(out):zeros() end
      if centroids then
        out = test_standarization(out, centroids[j])
      end
      result:axpy(1/#models, out)
    end
    result:clamp(0,1)
    --for j,id in ipairs(idxs) do
    -- Dog_1_test_segment_0001.mat,0
    f:write(names[id]:basename())
    f:write(".mat,")
    --f:write(result:get(j,1))
    f:write(result:get(1,1))
    f:write("\n")
    --end
  end
  f:close()
end

-- A into a file the validation prediction and target matrices.
function common.append(where, output, tgt, sampling_size, rnd)
  assert(output:dim(1) == tgt:dim(1))
  assert(#output:dim() == 2)
  assert(#output:dim() == #tgt:dim())
  if where then
    local m
    if not sampling_size or sampling_size == 0 then
      m = matrix.join(2, output, tgt)
    else
      print("# RESAMPLING VALIDATION = ", sampling_size)
      m = matrix(sampling_size, 2)
      local N = output:dim(1)
      for i=1,sampling_size do
        local j = rnd:randInt(1,N)
        local o,t = output:get(j,1),tgt:get(j,1)
        m:set(i,1,o):set(i,2,t)
      end
    end
    local f = io.open(where, "a")
    m:write(f, { tab=true })
    f:close()
  end
  local mv0,mv1 = stats.mean_var(),stats.mean_var()
  output:map(tgt, function(x,y) if y>0 then mv1:add(x) else mv0:add(x) end end)
  local a0,v0 = mv0:compute()
  local a1,v1 = mv1:compute()
  print("# A0: ", a0, v0)
  print("# A1: ", a1, v1)
  return a0,a1,v0,v1
end

function common.loss_stuff(loss)
  local log_scale,smooth,loss_param
  if loss == "batch_fmeasure_micro_avg" or loss == "batch_fmeasure_macro_avg" then
    smooth = false
    loss_param={beta=1.0}
  else smooth = true
  end
  if loss == "cross_entropy" then
    log_scale = true
  end
  return log_scale,smooth,loss_param
end

local function compute_pca(ds,by_rows,save_matrix)
  local m
  if not by_rows then
    m = matrix.col_major(ds:numPatterns(), ds:patternSize())
    for ipat,pat in ds:patterns() do
      local pat = pat
      m:select(1,ipat):copy( pat:rewrap(pat:size()) )
    end
  else
    local dims = ds:getPattern(1):dim()
    assert(#dims == 3)
    m = matrix.col_major(ds:numPatterns()*dims[1],dims[2]*dims[3])
    local k=0
    for ipat,pat in ds:patterns() do
      local pat = pat
      for i=1,dims[1] do
        k=k+1
        m:select(1,k):copy( pat:select(1,i):contiguous():
                              rewrap(dims[2]*dims[3]) )
      end
    end
  end
  if save_matrix then
    m:toTabFilename(save_matrix)
  end
  local U,S,VT = stats.pca(m)
  local takeN,eigen_value,prob_mass = stats.pca_threshold(S,0.99)
  return U,S,VT,takeN,eigen_value,prob_mass
end
-- make it callable from outside this module
common.compute_pca = compute_pca

local function push_whitening_component(the_net, wh_component, ds, by_rows, takeN)
  if not by_rows then
    the_net:push(wh_component)
    return wh_component:get_output_size()
  else
    local dims = ds:getPattern(1):dim()
    local takeN = takeN or dims[2]*dims[3]
    the_net:push( ann.components.rewrap{ size=dims } )
    the_net:push( ann.components.copy{ times = dims[1], input = dims[1]*dims[2]*dims[3] } )
    local join = ann.components.join()
    for i=1,dims[1] do
      local stack = ann.components.stack()
      stack:push( ann.components.select{ dimension=1, index=i }  )
      stack:push( ann.components.flatten() )
      stack:push( wh_component:clone() )
      stack:build{ input = dims[1]*dims[2]*dims[3], output = takeN }
      join:add( stack )
    end
    the_net:push( join )
    return dims[1]*takeN
  end
end

function common.push_zca_whitening(the_net, ds, by_rows)
  print("# COMPUTING ZCA")
  local U,S,VT,takeN,eigen_value,prob_mass = compute_pca(ds,by_rows)
  local epsilon = by_rows and 0.01 or 0.1
  print("#      PCA 0.99 IN SAMPLE ", takeN, U:dim(1))
  print("#      PCA 0.99 PROB      ", prob_mass)
  print("#      PCA 0.99 EIG       ", eigen_value)
  print("#      EPSILON            ", epsilon)
  return push_whitening_component(the_net,
                                  ann.components.zca_whitening{ U=U, S=S, epsilon=epsilon, takeN = takeN, },
                                  ds, by_rows)
end

function common.push_pca_reduction(the_net, ds, by_rows)
  print("# COMPUTING PCA")
  local U,S,VT,takeN,eigen_value,prob_mass = compute_pca(ds,by_rows)
  local epsilon = by_rows and 0.01 or 0.1
  print("#      PCA 0.99 IN SAMPLE ", takeN, U:dim(1))
  print("#      PCA 0.99 PROB      ", prob_mass)
  print("#      PCA 0.99 EIG       ", eigen_value)
  print("#      EPSILON            ", epsilon)
  return push_whitening_component(the_net,
                                  ann.components.pca_whitening{ U=U, S=S, epsilon=epsilon, takeN = takeN, },
                                  ds, by_rows, takeN)
end

function common.build_mlp_extractor(params)
  local dims = params.train_ds:getPattern(1):dim()
  local isize = params.input
  local the_net = ann.components.stack():push( ann.components.flatten() )
  if params.zca then
    isize = common.push_zca_whitening(the_net, params.train_ds, params.by_rows)
  elseif params.pca then
    isize = common.push_pca_reduction(the_net, params.train_ds, params.by_rows)
  end
  for i,hsize in ipairs( params.layers ) do
    if hsize == 0 then break end
    the_net:
      push( ann.components.hyperplane{ input=isize, output=hsize, bias_weights="b"..i, dot_product_weights="w"..i, } ):
      push( ann.components.actf[params.actf]() )
    isize = hsize
    if params.dropout > 0 then
      print("# DROPOUT LAYER = ", i)
      the_net:push( ann.components.dropout{ random=params.perturbation_random,
                                            prob=params.dropout } )
    end
  end
  return the_net,isize
end

function common.build_mlp_extractor_EM(params)
  local isize = params.input
  local the_net = ann.components.stack():push( ann.components.flatten() )
  if params.zca then
    isize = common.push_zca_whitening(the_net, params.train_ds)
  elseif params.pca then
    isize = common.push_pca_reduction(the_net, params.train_ds)
  end
  for i,hsize in ipairs( params.layers ) do
    if hsize == 0 then break end
    print("# ADDING LAYER = ", i, "SIZE = ", hsize)
    the_net:
      push( ann.components.hyperplane{ input=isize, output=hsize, bias_weights="b"..i, dot_product_weights="w"..i, } ):
      push( ann.components.actf[params.actf]() )
    isize = hsize
    if params.dropout > 0 then
      print("# DROPOUT LAYER = ", i)
      the_net:push( ann.components.dropout{ random=params.perturbation_random,
                                            prob=params.dropout } )
    end
  end
  return the_net,isize
end

function common.push_logistic_layer(the_net, params)
  the_net:push( ann.components.hyperplane{ input=params.input,
                                           output=1,
                                           dot_product_weights="wN",
                                           bias_weights="bN" } )
  if params.log_scale then
    the_net:push( ann.components.actf.log_logistic() )
  else
    the_net:push( ann.components.actf.logistic() )
  end
end

-- ds must contain matrices of NxFxC where N is time dimension, F filters
-- dimension, C recording channels dimension
function common.correlation_dataset(ds)
  
end

-- stores cross validations blocks into a disk directory
function common.store_cross_validation_blocks(output_dir)
  os.execute("mkdir -p " .. output_dir)
  local data_0,data_1 = load_sequences(name2id)
  for _,subject in ipairs{ "Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5",
                           "Patient_1", "Patient_2" } do
    local blocks = compute_cross_validation_partition(subject, data_0, data_1)
    local f = assert( io.open("%s/%s"%{output_dir,subject}, "w") )
    f:write(iterator(ipairs(blocks)):
            map(function(k,t) return table.concat(t, " ") end):concat("\n"))
    f:write("\n")
    f:close()
  end
end

------------------
-- EM ALGORITHM --
------------------

local function windowed_input(m, context, shape)
  assert(m and context)
  local M = 2*context + 1
  local dim = m:dim()
  assert(#dim==2 or #dim==3 or #dim==4, "Incorrect dim size: %d"%{#dim})
  if #dim == 3 then
    local m = m:rewrap(dim[1],dim[2]*dim[3]):clone("row_major")
    return dataset.matrix(m, { patternSize = { M, m:dim(2) },
                               offset = { context, 0 },
                               numSteps = { m:dim(1) - M + 1, 1 }, })
  elseif #dim == 4 then
    local ds_tbl = {}
    for i=1,dim[1] do
      local m = m:select(1,i):contiguous()
      table.insert(ds_tbl, windowed_input(m, context))
    end
    return dataset.union(ds_tbl)
  else
    assert(shape)
    local ds_tbl = {}
    for i=1,dim[1] do
      local m = m:select(1,i):contiguous():rewrap(table.unpack(shape))
      table.insert(ds_tbl, windowed_input(m, context))
    end
    return dataset.union(ds_tbl)
  end
end

function common.windowed_dataset(input_ds, output_ds, context, shape)
  assert(input_ds and context)
  local M = 2*context + 1
  local dim = input_ds:getPattern(1):dim()
  assert(#dim == 3)
  local result_input_ds  = {}
  local result_output_ds
  for ipat,pat in input_ds:patterns() do
    local ds  = windowed_input(pat, context, shape)
    table.insert(result_input_ds, ds)
  end
  result_input_ds = dataset.union(result_input_ds)
  result_input_ds = dataset.token.wrapper(result_input_ds)
  if output_ds then
    local out_m = matrix(result_input_ds:numPatterns(),1)
    local k = 1
    for ipat,pat in output_ds:patterns() do
      assert(#pat == 1)
      for i=1,dim[1]-M+1 do out_m:set(k,1,pat[1]) k=k+1 end
    end
    result_output_ds = dataset.matrix(out_m)
    assert(result_output_ds:numPatterns() == result_input_ds:numPatterns())
  end
  return result_input_ds,result_output_ds
end

function common.forward_EM(trainer, input, context, shape)
  assert(trainer and input and context)
  local in_ds  = windowed_input(input,context,shape)
  local out_ds = trainer:use_dataset{
    input_dataset  = in_ds,
    output_dataset = dataset.matrix(matrix(in_ds:numPatterns(),1))
  }
  return out_ds:toMatrix()
end

function common.use_dataset_EM(trainer, val_data, context)
  assert(trainer and val_data and context and val_data.input_dataset and val_data.output_dataset)
  local net = trainer:get_component()
  for ipat,input in val_data.input_dataset:patterns() do
    local target = val_data.output_dataset:getPattern(ipat)
    assert(#target == 1)
    local target = matrix(1,1,target)
    local output = common.forward_EM(trainer, input, context)
    local activation = output:max(1)
    val_data.output_dataset:putPattern(ipat, { activation:get(1,1) })
  end
  return val_data.output_dataset
end

function common.validate_EM(trainer, val_data, context)
  assert(trainer and val_data and context)
  local net = trainer:get_component()
  local loss = val_data.loss or trainer:get_loss_function()
  loss:reset()
  for ipat,input in val_data.input_dataset:patterns() do
    local target = val_data.output_dataset:getPattern(ipat)
    assert(#target == 1)
    local target = matrix(1,1,target)
    local output = common.forward_EM(trainer, input, context)
    local activation = output:max(1)
    loss:accum_loss(loss:compute_loss(activation, target))
  end
  return loss:get_accum_loss()
end

function common.expectation(trainer, input_ds, output_ds, TH, log_scale)
  local orig_out_m = output_ds:toMatrix()
  local output_ds  = trainer:use_dataset{
    input_dataset  = input_ds,
    output_dataset = output_ds,
  }
  local m = output_ds:toMatrix()
  if log_scale then m:exp() end
  print("# MAX ACTIVATION = ", (m:max()))
  m:copy(m:gt(TH):to_float()):cmul(orig_out_m)
  local result_positives = m:sum()
  print("# ORIG POSITIVES =   ", orig_out_m:sum())
  print("# REMOVE SAMPLES =   ", (orig_out_m - m):sum())
  print("# RESULT POSITIVES = ", result_positives)
  return result_positives > 0 and dataset.matrix(m) or false
end

return common
