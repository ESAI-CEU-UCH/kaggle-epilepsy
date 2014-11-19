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

local common = {}

-- load environment configuration into commont table
iterator(io.lines("scripts/env.sh")):
apply(function(line)
      local k=line:match("^export ([^%s]+)%s*=%s*[^%s]+$")
      if k then
        local v=april_assert(os.getenv(k), "%s %s %s",
                             "Unable to load environment variables, please check",
                             "that scripts/conf.sh has been loaded by using:",
                             ". scripts/env.sh")
        common[k] = v
      end
end)

-- libraries import
local mop   = matrix.op
local amean = stats.amean
local gmean = stats.gmean
local hmean = stats.hmean

-- check APRIL-ANN version
local major,minor,commit = util.version()
assert( tonumber(major) >= 0 and tonumber(minor) >= 4,
        "Incorrect APRIL-ANN version" )

------------------------------------------------------------------------------
---------------------- CROSS VALIDATION PARTITIONS ---------------------------
------------------------------------------------------------------------------

-- load all the sequence numbers taken from original mat files and returns a
-- data_0,data_1 tables of pairs {filename,sequence_number}
local function load_sequences(filtered_names)
  local f = assert( io.open(common.SEQUENCES_PATH) )
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
  -- a table which associates sequence number with a list of filenames.
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
  -- Receives a table of blocks (empty or not), a table of sequences (as above),
  -- and the desired number of blocks. The result is stored at the given blocks
  -- reference (a table).
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
-- tr_data,va_data for each of the partitions. The classify_func will be used
-- to compute predictions of validation after training.
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
  local mv = stats.running.mean_var()
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

-- receives a matlab matrix filename and returns the EEG matrix, sampling
-- frequency, number of channles and sequence number
function common.load_matlab_file(filename)
  local ok,m,hz,num_channels,seq = xpcall(function()
      local data = assert( matlab.read(filename) )
      local name,tbl = assert( next(data) )
      local hz = assert( tbl.sampling_frequency:get(1,1) )
      local seq = tbl.sequence:get(1,1)
      local m = assert( tbl.data:to_float() )
      return m,hz,m:dim(1),seq
                         end,
    debug.traceback)
  if not ok then
    print(m)
    error("An error happened processing %s"%{filename})
  end
  return m, hz, num_channels, seq
end

-- recieves a matrix with NxM, sampling_frequency, window size in seconds,
-- window advance in seconds, and returns an array with N matrices of TxF, where
-- N is the number of channels, M is the number of samples, T is the number of
-- window slices and F is the size of FFT output.
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

-- computes logarithm compression of a matrix
function common.compress(m)
  return m:clone():clamp(1.0, (m:max())):log()
end

-- returns a closure which can be used in a parallel_for_each in order to
-- preprocess (FFT computation) a list of matlab filenames. The closure returns
-- the sequence number of the file, in order to extract out this information for
-- cross-validation scheme.
function common.make_prep_function(HZ,FFT_SIZE,WSIZE,WADVANCE,out_dir,filter)
  return function(mat_filename)
    print("#",mat_filename)
    collectgarbage("collect")
    local m,hz,N,seq = common.load_matlab_file(mat_filename)
    -- sanity check
    assert( math.abs(hz - HZ) < 1 )
    -- fft_tbl is an array of N FFT matrices with size TxF, where N is the
    -- number of channels (subject dependent), T is the number temporal slices
    -- (depends in WSIZE, WADVANCE and the number of columns in m) and F is
    -- the FFT_SIZE.
    local fft_tbl = common.compute_fft(m, hz, WSIZE, WADVANCE)
    assert( #fft_tbl == N, "Incorrect number of channels" )
    -- for each channel
    for i=1,#fft_tbl do
      -- store every channel in an independent file
      local out_filename = "/%s.channel_%02d.csv.gz" %
        { (mat_filename:basename():gsub("%.mat", "" )), i, }
      -- sanity check
      assert( fft_tbl[i]:dim(2) == FFT_SIZE, fft_tbl[i]:dim(2) )
      local bf = filter( fft_tbl[i] )
      bf:toTabFilename(out_dir .. out_filename)
    end
    return seq
  end
end

-- returns a filter function prepared to transform an input matrix with FFT_SIZE
-- columns, recorded at HZ sampling rate
function common.compute_PLOS_filter(HZ, FFT_SIZE)
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
  local NUM_BF = #limits
  local filter = matrix(FFT_SIZE,NUM_BF):zeros()
  for i=1,NUM_BF do
    local ini = math.ceil(limits[i][1] / BIN_WIDTH)
    local fin = math.floor(limits[i][2] / BIN_WIDTH)
    local sz  = fin - ini + 1
    -- select a column of the filter matrix
    local bf  = filter:select(2,i)
    -- set the rows from (ini,fin) to 1/sz (arithmetic mean filter)
    bf({ ini , fin }):fill(1/sz)
  end
  -- convert the filter into a sparse matrix (improve computation time)
  local filter = matrix.sparse.csc(filter)
  -- returns a closure which receives a matrix, applies the matrix filter,
  -- log-compress the result and returns it.
  return function(m)
    local out = common.compress( m * filter )
    -- sanity checks
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

function common.load_data(list,params)
  print("# LOADING", list)
  local no_channels = params.no_channels
  local num_channels = params.channels
  local context = params.context
  local d1 = params.d1
  local d2 = params.d2
  local input_mat_tbl = {}
  local labels = {}
  local list_names = {}
  local ncols,nrows
  for path in io.lines(list) do
    table.insert(list_names, path:basename())
    local input
    local mat_tbl = {}
    if no_channels then
      local filename = "%s.txt"%{path}
      local f = april_assert(io.open(filename), "Unable to open %s", filename)
      local ok,msg = xpcall(function()
          mat_tbl[#mat_tbl + 1] = matrix.read(f,
                                              { [matrix.options.tab]   = true,
                                                [matrix.options.ncols] = ncols,
                                                [matrix.options.nrows] = nrows, })
                            end,
        debug.traceback)
      if not ok then
        print(msg)
        error("Error happened loading %s"%{filename})
      end
      nrows,ncols = table.unpack(mat_tbl[#mat_tbl]:dim())
      assert(#mat_tbl == 1)
      input = mat_tbl[#mat_tbl]
    else
      for j=1,num_channels do
        local filename = "%s.channel_%02d.csv.gz"%{path,j}
        local f = april_assert(io.open(filename), "Unable to open %s", filename)
        local ok,msg = xpcall(function()
            mat_tbl[#mat_tbl + 1] = matrix.read(f,
                                                { [matrix.options.tab]   = true,
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
      input = matrix.join(2,table.unpack(mat_tbl))
    end
    assert(input:dim(1) == nrows)
    assert(input:dim(2) == ncols * #mat_tbl)
    local label = get_label_of(path)
    table.insert(input_mat_tbl, input)
    if label then
      for i=1,nrows do labels[#labels + 1] = label end
    end
  end
  local input_mat = matrix.join(1, input_mat_tbl)
  assert(input_mat:dim(1) == #input_mat_tbl * nrows)
  if #labels > 0 then assert(#labels == input_mat:dim(1)) end
  local in_ds = dataset.matrix(input_mat)
  if params.cor then
    local cors = common.load_data(params.cor,{ list = params.cor,
                                               no_channels = true })
    assert(in_ds:numPatterns() == cors.input_dataset:numPatterns())
    in_ds = dataset.join{ in_ds, cors.input_dataset }
  end
  if context then
    in_ds = dataset.contextualizer(in_ds, context, context)
  end
  if d1 or d2 then
    in_ds = dataset.deriv{ dataset = in_ds,
                           deriv0  = true,
                           deriv1  = d1 or false,
                           deriv2  = d2 or false, }
  end
  return {
    input_dataset  = in_ds,
    output_dataset = (#labels > 0 and dataset.matrix(matrix(labels))) or nil,
  },
  list_names
end

function common.compute_means_devs(ds)
  return ds:mean_deviation()
end

function common.apply_std(ds,means,devs)
  print("# Applying standarization")
  return dataset.sub_and_div_normalization(ds,means,devs)
end

local comb_options = {
  max = function(slice,r) slice:max(1,r) end,
  amean = function(slice,r) slice:sum(1, r) r:scal(1/math.sqrt(slice:size())) end,
  gmean = function(slice,r) r:set(1,1, 1.0 - gmean( 1.0 - slice )) end,
  hmean = function(slice,r) r:set(1,1, 1.0 - hmean(1.0 - slice)) end,
}
function common.combine_filename_outputs(outputs, nrows, comb_name, target_check)
  assert(outputs:max() <= 1)
  assert(outputs:min() >= 0)
  local dim = outputs:dim()
  assert((#dim == 2) and (dim[1] % nrows == 0) and (dim[2] == 1))
  local result = matrix(dim[1]/nrows,1)
  local k=0
  for i=1,dim[1],nrows do
    k=k+1
    local slice = outputs({i, i + nrows - 1}, 1)
    local r = result(k,1)
    local comb_func = assert(comb_options[comb_name], "Incorrect combine name")
    comb_func(slice,r)
    assert(not target_check or slice:sum() == slice:size() or slice:sum() == 0)
  end
  return result
end

function common.save_test(output, names, ds, params, classify, ...)
  local nrows = ds:numPatterns()/#names
  assert(math.floor(nrows) == nrows)
  local f = assert(io.open(output, "a"))
  local ds = dataset.token.wrapper(ds)
  local g = assert(io.open("%svalidation_%s.test.txt"%{params.PREFIX,
                                                       params.SUBJECT}, "w"))
  for pat,indices in trainable.dataset_multiple_iterator{ bunch_size = nrows,
                                                          datasets = { ds }, } do
    assert(#indices == nrows)
    local id = (indices[1]-1)/nrows + 1
    local result = classify(pat:clone(), ...)
    if params.log_scale then result = result:clone():exp() end
    -- remove zero values
    -- result:cmul(result:clone():gt(1e-20))
    result = common.combine_filename_outputs(result, nrows, params.combine)
    result:clamp(0,1)
    assert(result:dim(1) == 1 and result:dim(2) == 1)
    f:write(names[id]:basename())
    f:write(".mat,")
    f:write(result:get(1,1))
    f:write("\n")
    --
    g:write(names[id]:basename())
    g:write(".mat,")
    g:write(result:get(1,1))
    g:write("\n")
    --
    --end
  end
  f:close()
  g:close()
end

-- A into a file the validation prediction and target matrices.
function common.append(where, output, tgt)
  assert(output:dim(1) == tgt:dim(1))
  assert(#output:dim() == 2)
  assert(#output:dim() == #tgt:dim())
  if where then
    local m
    m = matrix.join(2, output, tgt)
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
  else
    log_scale = false
  end
  return log_scale,smooth,loss_param
end

local function compute_pca(ds,by_rows,save_matrix)
  local m
  if not by_rows then
    m = matrix(ds:numPatterns(), ds:patternSize())
    for ipat,pat in ds:patterns() do
      m:select(1,ipat):copy( pat:rewrap(pat:size()) )
    end
  else
    local dims = ds:getPattern(1):dim()
    assert(#dims == 3)
    m = matrix(ds:numPatterns()*dims[1],dims[2]*dims[3])
    local k=0
    for ipat,pat in ds:patterns() do
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

function common.print_weights_norm2(trainer, name)
  local mapf = function(name,w) return "%7.3f"%{trainer:norm2(name)} end
  return iterator(trainer:iterate_weights(name)):map(mapf):concat(" ", " ")
end

function common.expectation(classify, input_ds, output_ds, TH, log_scale, ...)
  local orig_tgt_m = output_ds:toMatrix()
  local out_m = classify(input_ds:toMatrix(), ...)
  if log_scale then out_m:exp() end
  print("# MAX ACTIVATION = ", (out_m:max()))
  out_m:copy(out_m:gt(TH):to_float()):cmul(orig_tgt_m)
  local result_positives = out_m:sum()
  print("# ORIG POSITIVES =   ", orig_tgt_m:sum())
  print("# REMOVE SAMPLES =   ", (orig_tgt_m - out_m):sum())
  print("# RESULT POSITIVES = ", result_positives)
  return result_positives > 0 and dataset.matrix(out_m) or false
end

function common.validate_EM(classify, val_data, loss, nrows, log_scale, combine, ...)
  assert(classify and val_data and loss)
  assert(math.floor(nrows) == nrows)
  loss:reset()
  local in_ds = dataset.token.wrapper(val_data.input_dataset)
  local tgt_ds = val_data.output_dataset
  for in_pat,tgt_pat,indices in trainable.dataset_multiple_iterator{ bunch_size = nrows,
                                                                     datasets = { in_ds,
                                                                                  tgt_ds }, } do
    assert(#indices == nrows)
    local id = (indices[1]-1)/nrows + 1
    local result = classify(in_pat:clone(), ...)
    if log_scale then result = result:clone():exp() end
    -- remove zero values
    -- result:cmul(result:clone():gt(1e-20))
    result = common.combine_filename_outputs(result, nrows, combine):clamp(0,1)
    tgt_pat = common.combine_filename_outputs(tgt_pat:clone(),
                                              nrows, "max", true)
    assert(result:dim(1) == 1 and result:dim(2) == 1)
    if log_scale then result:log() end
    local l = table.pack(loss:compute_loss(result:clone(),
                                           tgt_pat:clone()))
    if l[1] then
      loss:accum_loss(table.unpack(l))
    end
  end
  return loss:get_accum_loss()
end

return common
