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
iterator(io.lines("settings.sh")):
apply(function(line)
      local k=line:match("^%s*export ([^%s]+)%s*=.*$")
      if k then
        local v=april_assert(os.getenv(k), "%s %s %s",
                             "Unable to load environment variables, please check",
                             "that scripts/conf.sh has been loaded by using:",
                             ". settings.sh")
        common[k] = v
      end
end)

common.SUBJECTS = common.SUBJECTS:tokenize("\n\r\t ")

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

-- Computes the cross validation partition for a given subject name (Dog_1,
-- Dog_2, ...) and the pair data_0,data_1 returned by load_sequences function.
-- It returns blocks table, which is an array with as many entries as blocks and
-- every entry is a list of filenames.
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
  
  -- rearrangue all 6 files in every sequence together
  local sequences_0 = divide_sequences(data_0)
  local sequences_1 = divide_sequences(data_1)
  
  -- compute number of blocks as the number of positive sequences with size 6
  local num_blocks = iterator(table.ivalues(sequences_1)):
  map(function(v) return #v end):
  filter(function(n) return n == 6 end):size()
  
  -- distribute the partition into blocks
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
  -- compute number of time slices (number of rows)
  local NROWS = all_train_data.input_dataset:numPatterns()/#list_names
  print("# CV INPUT NUM ROWS = ", NROWS)
  -- sanity check
  assert(math.floor(NROWS) == NROWS)
  -- dictionary of filename into a numeric position
  local name2id = table.invert(list_names)
  -- take the subject from 
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
  local outm = matrix.join(1,iterator(val_results):field(1):table())
  local tgtm = matrix.join(1,iterator(val_results):field(2):table())
  local val_result = matrix.join(2,outm,tgtm)
  val_result:toTabFilename("%s/validation_%s.txt"%{params.PREFIX,params.SUBJECT})
  return {roc_curve:compute_area()},{mv:compute()},val_result
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
      local seq = tbl.sequence
      if seq then seq = seq:get(1,1) end
      local m = assert( tbl.data:to_float() )
      return m,hz,m:dim(1),seq or -1
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
function common.make_prep_function(HZ,FFT_SIZE,WSIZE,WADVANCE,out_dir,
                                   filter,channels)
  return function(mat_filename)
    collectgarbage("collect")
    local exists = true
    for i=1,channels do
      local aux = "%s/%s.channel_%02d.csv.gz" %
        { out_dir, (mat_filename:basename():gsub("%.mat", "" )), i }
      if not common.exists(aux) then exists = false break end
    end
    if not exists then
      print("#",mat_filename)
      local m,hz,N,seq = common.load_matlab_file(mat_filename)
      -- sanity check
      assert( math.abs(hz - HZ) < 1 )
      -- fft_tbl is an array of N FFT matrices with size TxF, where N is the
      -- number of channels (subject dependent), T is the number temporal slices
      -- (depends in WSIZE, WADVANCE and the number of columns in m) and F is
      -- the FFT_SIZE.
      local fft_tbl = common.compute_fft(m, hz, WSIZE, WADVANCE)
      assert( #fft_tbl == N, "Incorrect number of channels" )
      assert( N == channels, "Incorrect number of channels" )
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
    else
      return -1
    end
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

-- get class label, 0 or 1 depending in the filename, or nil if not
-- preictal/interictal found in the filename
local function get_label_of(basename)
  if basename:match("preictal") then return 1
  elseif basename:match("interictal") then return 0
  else return nil
  end
end

local function protected_call(func, error_msg, ...)
  return func(...)
  -- local result = table.pack(xpcall(func(...), debug.traceback))
  -- local ok = table.remove(result,1)
  -- if not ok then
  --   error(result[1] .. "\n" .. error_msg)
  -- end
  -- return table.pack(result)
end

-- Loads a list of filenames from a given path using the indicated mask and
-- the params.subject parameter. The function returns input_dataset and
-- output_dataset if labels are available in the filenames (no test), and
-- the list of loaded filenames.
function common.load_data(path,mask,params)
  print("# LOADING", path, params.SUBJECT, mask)
  -- true if features don't are segmented into channels
  local no_channels = params.no_channels
  local num_channels
  -- context size (the time slice window is c+1+c)
  local context = params.context
  -- a list of input matrices
  local input_mat_tbl = {}
  -- a list with labels of every matrix row
  local labels = {}
  -- a list with all filenames
  local list_names = {}
  local ncols,nrows -- for sanity check
  local subject_mask = "%s%s"%{params.SUBJECT,mask}
  if not no_channels then
    subject_mask = subject_mask .. "channel_01*"
  end
  local cmd = "find %s -name %s | sort"%{path,subject_mask}
  for filename in iterator(io.popen(cmd):lines()) do
    local basename = filename:basename():gsub(".channel.*$",""):gsub(".txt","")
    table.insert(list_names, basename)
    local input
    local mat_tbl = {}
    if no_channels then -- load a matrix without channel splits
      local f = april_assert(io.open(filename), "Unable to open %s", filename)
      protected_call(function()
          mat_tbl[#mat_tbl + 1] = matrix.read(f,
                                              { [matrix.options.tab]   = true,
                                                [matrix.options.ncols] = ncols,
                                                [matrix.options.nrows] = nrows, })
                     end,
        "Error happened loading %s"%{filename})
      f:close()
      nrows,ncols = table.unpack(mat_tbl[#mat_tbl]:dim())
      assert(#mat_tbl == 1)
      input = mat_tbl[#mat_tbl]
    else -- load a matrix splitted into different channel filenames
      local list = glob(filename:gsub("channel_..", "channel_??"))
      num_channels = num_channels or #list
      assert(num_channels == #list, "Incorrect number of channels")
      for _,filename in ipairs(list) do
        local f = april_assert(io.open(filename), "Unable to open %s", filename)
        protected_call(function()
            mat_tbl[#mat_tbl + 1] = matrix.read(f,
                                                { [matrix.options.tab]   = true,
                                                  [matrix.options.ncols] = ncols,
                                                  [matrix.options.nrows] = nrows, })
                       end,
          "Error happened loading %s"%{filename})
        f:close()
        nrows,ncols = table.unpack(mat_tbl[#mat_tbl]:dim())
      end
      -- join by columns all channels
      input = matrix.join(2,table.unpack(mat_tbl))
    end
    -- sanity check
    assert(input:dim(1) == nrows)
    assert(input:dim(2) == ncols * #mat_tbl)
    local label = get_label_of(filename)
    -- annotate the input matrix
    table.insert(input_mat_tbl, input)
    if label then
      -- annotate as many labels as rows has the input matrix
      for i=1,nrows do labels[#labels + 1] = label end
    end
  end -- for path in list
  -- join by rows all the loaded matrices
  local input_mat = matrix.join(1, input_mat_tbl)
  -- sanity checks
  assert(input_mat:dim(1) == #input_mat_tbl * nrows)
  if #labels > 0 then assert(#labels == input_mat:dim(1)) end
  -- build a dataset with all the input matrix
  local in_ds = dataset.matrix(input_mat)
  if params.cor then
    -- if correlation features are available, load them recursively
    local cors = common.load_data(params.cor, mask, { no_channels = true,
                                                      SUBJECT = params.SUBJECT })
    assert(in_ds:numPatterns() == cors.input_dataset:numPatterns())
    -- join both set of features
    in_ds = dataset.join{ in_ds, cors.input_dataset }
  end
  if context then
    -- BUG: we are building contextualized input which mixes the rows in
    -- boundaries of the matrices, it is not an important BUG, but some noise
    -- may be introduced into training.
    in_ds = dataset.contextualizer(in_ds, context, context)
  end
  -- return a table with input and output datasets, and the list of loaded
  -- filenames
  return {
    input_dataset  = in_ds,
    output_dataset = (#labels > 0 and dataset.matrix(matrix(labels))) or nil,
  },
  list_names
end

-- compute mean and deviation from a given dataset
function common.compute_means_devs(ds)
  return ds:mean_deviation()
end

-- apply zero-mean one-variance transformation given a dataset and mean/devs
-- tables.
function common.apply_std(ds,means,devs)
  print("# Applying standarization")
  return dataset.sub_and_div_normalization(ds,means,devs)
end

-- different possible combinations of every probability computed for every file
local comb_options = {
  -- just take the maximum
  max = function(slice,r) slice:max(1,r) end,
  -- a special version of arithmetic mean
  amean = function(slice,r) slice:sum(1, r) r:scal(1/math.sqrt(slice:size())) end,
  -- geometric mean of the complement probabilities
  gmean = function(slice,r) r:set(1,1, 1.0 - gmean( 1.0 - slice )) end,
  -- harmonic mean of complement probabilities
  hmean = function(slice,r) r:set(1,1, 1.0 - hmean(1.0 - slice)) end,
}
function common.combine_filename_outputs(outputs, nrows, comb_name, target_check)
  assert(outputs:max() <= 1)
  assert(outputs:min() >= 0)
  local dim = outputs:dim()
  -- sanity check
  assert((#dim == 2) and (dim[1] % nrows == 0) and (dim[2] == 1))
  -- result matrix, after combine all the given outputs
  local result = matrix(dim[1]/nrows,1)
  local k=0
  for i=1,dim[1],nrows do
    k=k+1
    local slice = outputs({i, i + nrows - 1}, 1) -- slice of the output matrix
    local r = result(k,1) -- result component
    local comb_func = assert(comb_options[comb_name], "Incorrect combine name")
    comb_func(slice,r) -- compute combination and store result at r
    -- sanity check for target matrices
    assert(not target_check or slice:sum() == slice:size() or slice:sum() == 0)
  end
  return result
end

-- save test predictions given the output filename, the test filenames list,
-- the dataset with all data, the classify function and its arguments.
function common.save_test(output, names, ds, params, classify, ...)
  local nrows = ds:numPatterns()/#names
  assert(math.floor(nrows) == nrows)
  local f = assert(io.open(output, "a"))
  local ds = dataset.token.wrapper(ds)
  -- auxiliar output file (for system combination)
  local g = assert(io.open("%s/validation_%s.test.txt"%{params.PREFIX,
                                                        params.SUBJECT}, "w"))
  -- for every possible file
  for pat,indices in trainable.dataset_multiple_iterator{ bunch_size = nrows,
                                                          datasets = { ds }, } do
    assert(#indices == nrows)
    local id = (indices[1]-1)/nrows + 1 -- index of the file
    local result = classify(pat:clone(), ...) -- compute result
    if params.log_scale then result = result:clone():exp() end -- scale it
    -- combine all the outputs
    result = common.combine_filename_outputs(result, nrows, params.combine)
    result:clamp(0,1)
    -- sanity check
    assert(result:dim(1) == 1 and result:dim(2) == 1)
    -- write the output into files f and g
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

-- returns properties of the given loss function name: log_scale, smooth,
-- loss_param table
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

function common.build_mlp_extractor(params)
  local isize = params.input
  local the_net = ann.components.stack():push( ann.components.flatten() )
  for i,hsize in ipairs( params.layers ) do
    if hsize == 0 then break end
    print("# ADDING LAYER = ", i, "SIZE = ", hsize)
    the_net:
      push( ann.components.hyperplane{ input=isize, output=hsize,
                                       bias_weights="b"..i,
                                       dot_product_weights="w"..i, } ):
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
  for _,subject in ipairs( common.SUBJECTS ) do
    local blocks = compute_cross_validation_partition(subject, data_0, data_1)
    local f = assert( io.open("%s/%s"%{output_dir,subject}, "w") )
    f:write(iterator(ipairs(blocks)):
            map(function(k,t) return table.concat(t, " ") end):concat("\n"))
    f:write("\n")
    f:close()
  end
end

-- print the max norm2 of all the weights matrices
function common.print_weights_norm2(trainer, name)
  local mapf = function(name,w) return "%7.3f"%{trainer:norm2(name)} end
  return iterator(trainer:iterate_weights(name)):map(mapf):concat(" ", " ")
end

-- return validation loss given classify function, val_data table, loss
-- function, number of rows per file, log_scale parameter, combine method and
-- extra arguments for classify function
function common.validate(classify, val_data, loss, nrows, log_scale, combine, ...)
  assert(classify and val_data and loss)
  assert(math.floor(nrows) == nrows)
  loss:reset()
  local in_ds = dataset.token.wrapper(val_data.input_dataset)
  local tgt_ds = val_data.output_dataset
  -- for every file
  for in_pat,tgt_pat,indices in trainable.dataset_multiple_iterator{ bunch_size = nrows,
                                                                     datasets = { in_ds,
                                                                                  tgt_ds }, } do
    -- sanity check
    assert(#indices == nrows)
    local id = (indices[1]-1)/nrows + 1 -- compute index of the file
    local result = classify(in_pat:clone(), ...) -- compute output
    if log_scale then result = result:clone():exp() end -- scale output
    -- combine output
    result = common.combine_filename_outputs(result, nrows, combine):clamp(0,1)
    -- combine target by using 'max' combination
    tgt_pat = common.combine_filename_outputs(tgt_pat:clone(),
                                              nrows, "max", true)
    -- sanity check
    assert(result:dim(1) == 1 and result:dim(2) == 1)
    if log_scale then result:log() end -- rescale again, for the loss function
    -- compute loss
    local l = table.pack(loss:compute_loss(result:clone(),
                                           tgt_pat:clone()))
    if l[1] then
      loss:accum_loss(table.unpack(l))
    end
  end
  return loss:get_accum_loss()
end

function common.exists(filename)
  local f = io.open(filename)
  if not f then return false else f:close() return true end
end

return common
